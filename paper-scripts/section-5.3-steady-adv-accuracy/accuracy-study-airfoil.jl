module AccuracyStudyAirfoil

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
diss_coeff = 0.25
deg = [1;2;3;4]
num1d = [4; 8; 16; 32]
num_sample = 1
pert_size = 0.25

# all boundaries have characteristic BCs applied to them
bc_map = Dict{Any,String}("ib" => "upwind")
for di = 1:Dim 
    bc_map[di*2-1] = "upwind"
    bc_map[di*2] = "upwind"
end

function uexact(x::AbstractVector{Float64})
    return exp(x[1] + x[2])
end
function uexact(x::AbstractMatrix{Float64})
    return exp.(x[1,:] + x[2,:])
end
function dudx(x)
    return Dim.*uexact(x) #exp(x[1] + 10*x[2] + 10*x[3])
end

"""
    ls = levset(x)

Returns the level-set value for an airfoil-shaped level-set function.
"""
function levset(x)
    return (x[1] - 1)^2 + (x[1] - 1)^3 - 16*x[2]^2
end
function levset_grad!(g, x)
    g[1] = 2*(x[1] - 1) + 3*(x[1] - 1)^2
    g[2] = -32*x[2]
    return nothing
end

"""
    xc, num_nodes = get_points(nx, ny, lower, upper, phi [, pert=0.25])

Generate quasi-uniform points by perturbing a uniform grid.  `nx` and `ny` are 
the number of points in x and y, respectively.  `lower` and `upper` are the 
bounds on the ranges for the box, and `phi` is the level-set function used to 
determine which points are inside and which are outside the domain.  `pert` 
determines the factor by which the random perturbation is applied as a fraction 
of the nodes spacing.
"""
function get_points(nx::Int, ny::Int, lower, upper, phi::Function; 
                    pert::Float64=0.25)
    xcart = zeros(Dim, nx*ny)
    xnd = reshape(xcart, (Dim, nx, ny))
    dx = [(upper[1] - lower[1])/nx;
          (upper[2] - lower[2])/ny]
    for I in CartesianIndices(xnd)
       # I[1] is the coordinate, so I[I[1] + 1] is the index for that coord
       xnd[I] = (I[I[1]+1] - 0.5)*dx[I[1]] + lower[I[1]]
       xnd[I] += pert*(2*rand()-1)*dx[I[1]]
    end
    num_nodes = 0
    xc = zeros(Dim,0)
    for i in axes(xcart,2)
        if phi(xcart[:,i]) > 0
            # lots of allocations here
            xc = hcat(xc, xcart[:,i])
            num_nodes += 1
        end
    end
    return xc, num_nodes
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
    A += diss_coeff*(diss.R_left' - diss.R_right')*spdiagm(diss.w_face)*(diss.R_left - diss.R_right)
    
    # finally, add the MMS terms
    b += sbp.H .* dudx(xc)
    
    return A, b
end

exact_vol = 2/15
L2err = zeros(length(num1d), num_sample, length(deg))
dx = zero(L2err)
Hdx = zero(L2err)
minH = zero(L2err)
maxerr = zero(L2err)

thick = 0.2
origin = SVector((0.0, -thick/2))
widths = SVector((1.0, thick))

for k = 1:num_sample 
    for (i, nx) in enumerate(num1d)
        # generate mesh for this sample
        xc, num_nodes = get_points(5*nx, nx, origin,
                                   origin + widths, levset, pert=pert_size)
        dx[i, k, :] .= 1/nx
        min_widths = [1/(5*nx), thick/nx]
        min_widths .*= 0.1
        mesh = CloudSBP.build_mesh(xc, widths, levset, min_widths,
                                   origin=origin)
        H_tol = ones(num_nodes)
        H_tol .*= prod(min_widths)*10

        for (dindex, degree) in enumerate(deg)

            println()
            println(repeat("=",80))

            CloudSBP.build_cell_stencils!(mesh, xc, 2*degree - 1)
            max_stencil, avg_stencil = CloudSBP.stencil_stats(mesh)
            println("degree = ",degree,": max_stencil = ",max_stencil,
                    ": avg_stencil = ",avg_stencil)

            m = CloudSBP.calc_moments!(mesh.root, levset, max(2,2*degree-1), 
                                       2*degree-1)
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

            Hvol = sum(H)
            println("Volume error = ",abs(Hvol - exact_vol))

            @assert( minH[i, k, dindex] > 0.0 )
            sbp = CloudSBP.build_first_derivative(mesh, bc_map, xc, levset, 
                                                  levset_grad!, degree, 
                                                  fit_degree=2*degree-1)
            diss = CloudSBP.build_face_dissipation(mesh.ifaces, xc, degree, 
                                                   levset, 
                                                   fit_degree=2*degree-1)

            A, b = build_system(sbp, diss, xc)

            # solve and compute error 
            println("size(A) = ",size(A))
            u = A\b 
            du = u - uexact(xc)
            L2err[i, k, dindex] = sqrt(dot(du,sbp.H.*du))
            dx[i, k, dindex] = 1/nx
            Hdx[i, k, dindex] = mean(H)^(1/Dim)
            maxerr[i, k, dindex] = maximum(abs.(du))
            println("degree = ",degree,": num_nodes = ",num_nodes,
                    ": L2 error = ", L2err[i, k, dindex],
                    ": max error = ", maxerr[i, k, dindex])

        end
    end
end

# write the error to file for plotting
f = open("error-airfoil.dat", "w")
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

end # module 