module TimingStudy
# Used to compile timing data for the major steps in the discretization process

using CloudSBP
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using SparseArrays

# For repeatable "random" tests
Random.seed!(42)

# study parameters
Dim = 2
vel = ones(Dim)
diss_coeff = 0.25
deg = [1;2;3;4]
num1d = [4; 8; 16; 32; 64]
num_sample = 11 # discard the first sample!
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
    return Dim.*uexact(x) 
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

points_time = zeros(length(num1d), num_sample, length(deg))
nodes = zero(points_time)
mesh_time = zero(points_time)
stencil_time = zero(points_time)
moments_time = zero(points_time)
lp_time = zero(points_time)
deriv_time = zero(points_time)
diss_time = zero(points_time)
system_time = zero(points_time)
solve_time = zero(points_time)

thick = 0.2
origin = SVector((0.0, -thick/2))
widths = SVector((1.0, thick))

for k = 1:num_sample 
    for (i, nx) in enumerate(num1d)
        # generate mesh for this sample
        stats = @timed xc, num_nodes = get_points(5*nx, nx, origin,
            origin + widths, levset, pert=pert_size)
        points_time[i,k,:] .= stats.time
        nodes[i,k,:] .= num_nodes

        min_widths = [1/(5*nx), thick/nx]
        min_widths .*= 0.1
        stats = @timed mesh = CloudSBP.build_mesh(xc, widths, levset, 
            min_widths, origin=origin)
        mesh_time[i,k,:] .= stats.time

        H_tol = ones(num_nodes)
        H_tol .*= prod(min_widths)*10

        for (dindex, degree) in enumerate(deg)

            println()
            println(repeat("=",80))

            stats = @timed CloudSBP.build_cell_stencils!(mesh, xc, 2*degree - 1)
            stencil_time[i,k,dindex] = stats.time

            max_stencil, avg_stencil = CloudSBP.stencil_stats(mesh)
            println("degree = ",degree,": max_stencil = ",max_stencil,
                    ": avg_stencil = ",avg_stencil)

            stats = @timed m = CloudSBP.calc_moments!(mesh.root, levset,
                max(2,2*degree-1), 2*degree-1)
            moments_time[i,k,dindex] = stats.time

            # optimize
            stats = @timed H, success = CloudSBP.solve_norm!(mesh.root, xc, 
                2*degree-1, H_tol, verbose=true)
            lp_time[i,k,dindex] = stats.time

            println("minimum(H) = ", minimum(H))
            if !success
                println("The LP optimization failed")
                deriv_time[i,k,dindex] = NaN
                diss_time[i,k,dindex] = NaN
                system_time[i,k,dindex] = NaN
                solve_time[i,k,dindex] = NaN
                continue
            end

            Hvol = sum(H)
            println("Volume error = ",abs(Hvol - exact_vol))
            
            stats = @timed sbp = CloudSBP.build_first_derivative(mesh, bc_map, 
                xc, levset, levset_grad!, degree, fit_degree=2*degree-1) # 2)
            deriv_time[i,k,dindex] = stats.time

            stats = @timed diss = CloudSBP.build_face_dissipation(mesh.ifaces, 
                xc, degree, levset, fit_degree=2*degree-1) #2)
            diss_time[i,k,dindex] = stats.time

            stats = @timed A, b = build_system(sbp, diss, xc)
            system_time[i,k,dindex] = stats.time

            # solve
            println("size(A) = ",size(A))
            stats = @timed u = A\b 
            solve_time[i,k,dindex] = stats.time

            # check errors in case of regressions 
            du = u - uexact(xc)
            L2err = sqrt(dot(du,sbp.H.*du))
            maxerr = maximum(abs.(du))
            println("degree = ",degree,": num_nodes = ",num_nodes,
                    ": L2 error = ", L2err, ": max error = ", maxerr)

        end
    end
end

# write the timings to file for plotting
f = open("timing-airfoil.dat", "w")
for d in axes(deg,1)
    for k in 1:num_sample 
        for i in axes(num1d,1)
            print(f, nodes[i,k,d], " ")
        end
        print(f, "# number of nodes (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, points_time[i,k,d], " ")
        end
        print(f, "# point gen. time (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, mesh_time[i,k,d], " ")
        end
        print(f, "# mesh gen. time (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, stencil_time[i,k,d], " ")
        end
        print(f, "# stencil build time (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, moments_time[i,k,d], " ")
        end
        print(f, "# moment calc time (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, lp_time[i,k,d], " ")
        end
        print(f, "# LP solve time (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, deriv_time[i,k,d], " ")
        end
        print(f, "# deriv build time (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, diss_time[i,k,d], " ")
        end
        print(f, "# diss build time (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, system_time[i,k,d], " ")
        end
        print(f, "# system build time (p=$d)")
        println(f)
        for i in axes(num1d,1)
            print(f, solve_time[i,k,d], " ")
        end
        print(f, "# system solve time (p=$d)")
        println(f)
    end
end
close(f)

end # module 