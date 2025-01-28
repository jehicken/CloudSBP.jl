module AccuracyStudyBoxCircle

using CloudSBP
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using Statistics

# For repeatable "random" tests
Random.seed!(42)

# study parameters
Dim = 2
vel = ones(Dim)
deg = [1;2;3;4]
num1d = [10; 20; 40; 80; 160]
num_sample = 10
pert_size = 0.25

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

function functional(x)
    xtrans = x .- [0.5; 0.5]
    r = norm(xtrans)
    theta = atan(xtrans[2], xtrans[1])
    return cos(2*theta)/r
end
fun_exact = 0.0

radius = 0.25
levset = x -> norm(x .- SVector(ntuple(i -> 0.5, Dim)))^2 - radius^2
function levset_grad!(g, x)
    g[:] = 2.0*x .- SVector(ntuple(i -> 1.0, Dim))
    return nothing
end

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

L2err = zeros(length(num1d), num_sample, length(deg))
dx = zero(L2err)
Hdx = zero(L2err)
minH = zero(L2err)
funerr = zero(L2err)

origin = SVector(ntuple(i -> 0.0, Dim))
widths = SVector(ntuple(i -> 1.0, Dim))

for k = 1:num_sample 
    for (i, nx) in enumerate(num1d)
        # generate mesh for this sample
        xc, num_nodes = get_points(nx, levset, pert=pert_size)
        dx[i, k, :] .= 1/nx
        min_widths = ones(Dim)/(2*nx)
        mesh = CloudSBP.build_mesh(xc, widths, levset, min_widths,
                                   origin=origin)
        H_tol = ones(num_nodes)
        vol = 1.0 - pi*0.25^2
        H_tol .*= 0.1*vol/num_nodes

        for (dindex, degree) in enumerate(deg)

            println()
            println(repeat("=",80))

            CloudSBP.build_cell_stencils!(mesh, xc, 2*degree - 1)
            max_stencil, avg_stencil = CloudSBP.stencil_stats(mesh)
            println("degree = ",degree,": max_stencil = ",max_stencil,
                    ": avg_stencil = ",avg_stencil)

            m = CloudSBP.calc_moments!(mesh.root, levset, max(2,2*degree-1),
                                       2)
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
f = open("fun-error-box-circle.dat", "w")
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


end