module SuccessRate
# This module is used study how often the LP fails to find a positive norm.

using CloudSBP
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random

# For repeatable "random" tests
Random.seed!(42)

# study parameters
Dim = 2
vel = ones(Dim)
deg = [1;2;3;4]
num1d = [8; 16; 32; 64]
pert_size = 0.25
num_sample = 1000 # how many different geometries to consider 
tau_factor = 0.25 # 0.000025 # 0.0025 # scales the tolerance for the norm 

# at least min_nodes must be available
min_nodes = binomial(Dim + 2*deg[end]-1, Dim) + 1

"""
    xc, num_nodes = get_points(nx, phi [, pert=0.25])

Generate quasi-uniform points by perturbing a uniform grid
"""
function get_points(nx, phi; pert::Float64=0.25)
    xcart = zeros(Dim, nx^Dim)
    xnd = reshape(xcart, (Dim, ntuple(i -> nx, Dim)...))
    dx = 1/nx
    for I in CartesianIndices(xnd)
        # I[1] is the coordinate, so I[I[1] + 1] is the index for that coord
        xnd[I] = (I[I[1]+1] - 1)/nx + 0.5*dx
        xnd[I] += pert*(2*rand()-1)*dx
    end
    # shift into [-1,1]^2
    xcart .*= 2.0
    xcart .-= 1.0
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

failures = zeros(Int, (num_sample, length(num1d), length(deg)))
dx = zeros(num_sample, length(num1d), length(deg))
min_dx = zero(dx)
min_H = zero(dx)
aconic = zero(dx)
bconic = zero(dx)

f = open("fail-geos.dat", "w")

for k = 1:num_sample

    # generate the level-set
    a = ((0.99-0.01)*rand() + 0.01)*rand([-1,1])
    b = ((0.99-0.01)*rand() + 0.01)
    aconic[k,:,:] .= a 
    bconic[k,:,:] .= b
    function levset(x)
        return 1.0 - x[1]^2/a - x[2]^2/b
    end
    function levset_grad!(g, x)
        g[:] = [-2.0*x[1]/a; -2.0*x[2]/b]
        return nothing
    end

    for (i, nx) in enumerate(num1d)
        # generate nodes for this sample and find min spacing
        xc, num_nodes = get_points(nx, levset, pert=pert_size)
        new = 0
        while num_nodes < min_nodes
            new += 1
            println("Too few nodes; regenerating with nx = ",nx + new)
            xc, num_nodes = get_points(nx+new, levset, pert=pert_size)
        end

        dx_min = 1e100 
        for j1 in 1:size(xc,2)
            for j2 = j1+1:size(xc,2) 
                dist = norm(xc[:,j1] - xc[:,j2])
                dx_min = min(dx_min, dist)
            end
        end
        H_tol = tau_factor*ones(num_nodes)*(2/nx)^2
        dx[k, i, :] .= 2.0/(nx+new) # nominal spacing
        min_dx[k, i, :] .= dx_min # min spacing

        for (dindex, degree) in enumerate(deg)

            root = Cell(SVector(ntuple(i -> -1.0, Dim)),
                    SVector(ntuple(i -> 2.0, Dim)),
                    CellData(Vector{Int}(), Vector{Int}()))
            
            # refine mesh, build stencil, get face lists
            CloudSBP.refine_on_points!(root, xc)
            CloudSBP.refine_on_levelset!(root, xc, levset, ones(Dim)/(2*nx))
            CloudSBP.mark_cut_cells!(root, levset)
            CloudSBP.build_nn_stencils!(root, xc, 2*degree-1)
   
            max_stencil = 0
            avg_stencil = 0
            count = 0
            for cell in allleaves(root)
                if CloudSBP.is_immersed(cell)
                    continue
                end
                count += 1
                max_stencil = max(max_stencil, length(cell.data.points))
                avg_stencil += length(cell.data.points)
            end
            avg_stencil = avg_stencil/count
            println()
            println(repeat("=",80))
            println("degree = ",degree,": max_stencil = ",max_stencil,
                    ": avg_stencil = ",avg_stencil)
            
            CloudSBP.set_xref_and_dx!(root, xc)          
            m = CloudSBP.calc_moments!(root, levset, max(2,2*degree-1), 2)
            
            # optimize
            H, success = CloudSBP.solve_norm!(root, xc, 2*degree-1, H_tol,
                                            verbose=true)
            if !success
                # The LP optimization failed; record the failure, save the 
                # mesh points and coefficients
                failures[k,i,dindex] = 1
                CloudSBP.points_vtk(xc, filename="fail-points-$k-$i-$(degree).vtu")                
                println(f, a, " ", b)
            end
            
            min_H[k,i,dindex] = minimum(H)
            println("minimum(H) = ", min_H[k,i,dindex])
                    
        end
    end
end
close(f)

# write the failure data to file
f = open("failures.dat", "w")
for k in 1:num_sample
    for i in axes(num1d,1)
        for d in axes(deg,1)
            print(f, failures[k,i,d], " ")
        end
        println(f)
        for d in axes(deg,1)
            print(f, dx[k,i,d], " ")
        end
        println(f)
        for d in axes(deg,1)
            print(f, min_dx[k,i,d], " ")
        end
        println(f)
        for d in axes(deg,1)
            print(f, min_H[k,i,d], " ")
        end
        println(f)
        for d in axes(deg,1)
            print(f, aconic[k,i,d], " ")
        end
        println(f)
        for d in axes(deg,1)
            print(f, bconic[k,i,d], " ")
        end
        println(f)
    end
end
close(f)

# make a Latex table of failures
suc_rate = round.(100 .* (num_sample .- sum(failures,dims=1))./num_sample, digits=2)
suc_rate = reshape(suc_rate, (length(num1d), length(deg)))
for i in axes(num1d,1)
    println(" $(num1d[i])  & $(suc_rate[i,1]) & $(suc_rate[i,2]) & $(suc_rate[i,3]) & $(suc_rate[i,4]) \\\\")
end

end # module
