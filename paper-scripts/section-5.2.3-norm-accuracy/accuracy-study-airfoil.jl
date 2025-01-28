module AccuracyStudyAirfoil

using CloudSBP
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using SpecialFunctions
using Statistics

# For repeatable "random" tests
Random.seed!(42)

# study parameters
Dim = 2
vel = ones(Dim)
diss_coeff = 0.25
deg = [1;2;3;4]
num1d = [4; 8; 16; 32; 64]
num_sample = 10
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
function functional(x)
    return exp(x[1])
end
fun_exact = 0.5*(-(5/4)*sqrt(pi)*erfi(1) + 0.5*exp(1)*3)

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

exact_vol = 2/15
L2err = zeros(length(num1d), num_sample, length(deg))
dx = zero(L2err)
Hdx = zero(L2err)
minH = zero(L2err)
funerr = zero(L2err)

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
                funerr[i, k, dindex] = NaN
                continue
            end
            
            fun = 0.0
            for i in axes(xc, 2)
                fun += functional(xc[:,i])*H[i]
            end
            funerr[i, k, dindex] = abs(fun - fun_exact)
            println("fun error = ",funerr[i, k, dindex])
            Hdx[i, k, dindex] = mean(H)^(1/Dim)

        end
    end
end

# write the error to file for plotting
f = open("fun-error-airfoil.dat", "w")
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
            print(f, funerr[i,k,d], " ")
        end
        println(f)
    end
end
close(f)

end # module 