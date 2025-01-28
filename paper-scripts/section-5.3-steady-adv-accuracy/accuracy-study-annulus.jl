module AccuracyStudyAnnulus

using CloudSBP
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using SparseArrays
using Statistics

# For repeatable "random" tests
Random.seed!(42)

# study parameters
Dim = 2
vel = ones(Dim)
deg = [1;2;3;4]
num1d = [6; 12; 24; 48]
num_sample = 10
pert_size = 0.25
growth = 0.1 # 4.0 for non-uniform, 0.1 for approximately uniform

# all boundaries have characteristic BCs applied to them
bc_map = Dict{Any,String}("ib" => "upwind")
for di = 1:Dim 
    bc_map[di*2-1] = "upwind"
    bc_map[di*2] = "upwind"
end

uexact(x::AbstractVector) = exp(sum(x))
dudx(x::AbstractVector) = Dim*exp(sum(x))
uexact(x::AbstractMatrix) = exp.(sum(x,dims=1))'
dudx(x::AbstractMatrix) = Dim*exp.(sum(x,dims=1))'

function levset(x)
    r = norm(x)
    return -(r^2 - 1)*(r^2 - 0.25)
end
function levset_grad!(g, x)
    r = norm(x)
    # dphidr is divided by r to avoid issues
    dphidr = -2*(r^2 - 0.25) - (r^2 - 1)*2
    g[:] = dphidr*x
    return nothing
end

"""
    xc, num_nodes = get_points(nr, nt[, pert=0.1/nx, growth=5.0])

Generate quasi-uniform points by perturbing a uniform grid
"""
function get_points(nr, nt; pert::Float64=0.1, growth::Float64=5.0)
    @assert(Dim == 2)
    radius(xi) = (exp(growth*xi) - 1.0)/(exp(growth) - 1.0)
    dradius(xi) = growth*exp(growth*xi)/(exp(growth)-1.0)
    rin = 0.5
    rout = 1.0
    xc = zeros(Dim, nr*nt)
    xnd = reshape(xc, (Dim, nr, nt))
    dtheta = 2*pi/nt
    H_tol = zeros(size(xc,2))
    Hnd = reshape(H_tol, (nr, nt))
    for i in axes(xnd,2) # r loop
        dr = dradius((i-1)/nr + 0.5/nr)/nr
        for j in axes(xnd,3) # theta loop
            xi = radius((i-1)/nr + 0.5/nr + pert*(2*rand()-1)/nr)
            r = (1-xi)*rin + xi*rout 
            theta = (j-1)*dtheta + 0.5*dtheta + pert*dtheta*(2*rand()-1)
            xnd[1,i,j] = r*cos(theta)
            xnd[2,i,j] = r*sin(theta)
            Hnd[i,j] = 0.1*dr*dtheta*r
        end
    end
    return xc, nr*nt, H_tol
end

"""
    A, b = build_system(sbp, diss, xc)

Returns the linear system based on the SBP discretization of the linear 
advection equation with velocity `vel` (global) and the geometry encoded in the 
SBP operator `sbp`.  The dissipation operator is `diss` and the nodes `xc` are 
included to impose boundary conditions and MMS terms by evaluating the exact 
solution (also global).
"""
function build_system(sbp, diss, xc)
    
    # Use the SBP operator to define the linear system
    num_nodes = length(sbp.H)
    A = similar(sbp.S[1] + sbp.S[1]')
    fill!(A, zero(Float64))
    b = zeros(num_nodes)
    
    # first apply the skew-symmetric part of the operator
    for di = 1:Dim 
        A += vel[di]*(sbp.S[di] - sbp.S[di]')
    end
    
    # now loop over the boundary
    for (bc_type, bndry) in sbp.E
        if bc_type == "upwind"
            for (xq, nrm, dof, prj) in zip(bndry.xq_face, bndry.nrm_face, 
                bndry.dof_face, bndry.prj_face)
                for i in axes(prj,2)
                    row = dof[i]
                    for j in axes(prj,2)
                        col = dof[j]
                        for q in axes(prj,1)
                            velnrm = dot(vel, nrm[:,q])
                            if velnrm < 0.0
                                # This is inflow 
                                A[row,col] -= 0.5*prj[q,i]*velnrm*prj[q,j]
                            else
                                # This is outflow 
                                A[row,col] += 0.5*prj[q,i]*velnrm*prj[q,j]
                            end
                        end
                    end                        
                    for q in axes(prj,1)
                        velnrm = dot(vel, nrm[:,q])
                        if velnrm < 0.0
                            # This is inflow
                            b[row] -= prj[q,i]*velnrm*uexact(xq[:,q])
                        end
                    end
                end
            end
        end
    end
    
    # apply the dissipation 
    A += 0.25*(diss.R_left' - diss.R_right')*spdiagm(diss.w_face)*(diss.R_left - diss.R_right)
    
    # finally, add the MMS terms
    b += sbp.H .* dudx(xc)
    
    return A, b
end

L2err = zeros(length(num1d), num_sample, length(deg))
dx = zero(L2err)
Hdx = zero(L2err)
minH = zero(L2err)
maxerr = zero(L2err)

origin = SVector(ntuple(i -> -1.0, Dim))
widths = SVector(ntuple(i -> 2.0, Dim))

for k = 1:num_sample 
    for (i, nx) in enumerate(num1d)
        # generate mesh for this sample 
        xc, num_nodes, H_tol = get_points(nx, 6*nx, pert=pert_size, 
                                          growth=growth)
        dx[i, k, :] .= 1/nx
        min_widths = ones(Dim)/(20*nx)
        mesh = CloudSBP.build_mesh(xc, widths, levset, min_widths,
                                    origin=origin)

        for (dindex, degree) in enumerate(deg)

            println()
            println(repeat("=",80))
            CloudSBP.build_cell_stencils!(mesh, xc, 2*degree - 1)
            max_stencil, avg_stencil = CloudSBP.stencil_stats(mesh)
            println("degree = ",degree,": max_stencil = ",max_stencil,
                    ": avg_stencil = ",avg_stencil)
            
            m = CloudSBP.calc_moments!(mesh.root, levset, max(2,2*degree-1), 2)

            # optimize
            H, success = CloudSBP.solve_norm!(mesh.root, xc, 2*degree-1, H_tol,
                                              verbose=true)
            minH[i, k, dindex] = minimum(H)
            println("minimum(H) = ", minH[i, k, dindex])

            if !success
                println("The LP optimization failed")
                Hdx[i, k, dindex] = NaN
                L2err[i, k, dindex] = NaN
                maxerr[i, k, dindex] = NaN
                continue
            end

            @assert( minH[i, k, dindex] > 0.0 )
            sbp = CloudSBP.build_first_derivative(mesh, bc_map, xc, levset, 
                                                  levset_grad!, degree,
                                                  fit_degree=2)
            diss = CloudSBP.build_face_dissipation(mesh.ifaces, xc, degree, 
                                                   levset, fit_degree=2)

            A, b = build_system(sbp, diss, xc)

            # solve and compute error 
            u = A\b 
            du = u - uexact(xc)
            L2err[i, k, dindex] = sqrt(dot(du,sbp.H.*du))
            Hdx[i, k, dindex] = mean(H)^(1/Dim)
            maxerr[i, k, dindex] = maximum(abs.(du))
            println("degree = ",degree,": num_nodes = ",num_nodes,": error = ",
                    L2err[i, k, dindex])

        end
    end
end

# write the error to file for plotting
f = open("error-annulus.dat", "w")
for d in axes(deg,1)
    for k in 1:num_sample 
        for i in axes(num1d,1)
            print(f, dx[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, Hdx[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, minH[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, L2err[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, maxerr[i,k,d], " ")
        end
        println(f)
    end
end
close(f)

end