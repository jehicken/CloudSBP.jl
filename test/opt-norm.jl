module OptNorm

using CutDGD
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using LevelSets
using CxxWrap
using SparseArrays
#using NearestNeighbors

# Following StaticArrays approach of using repeatable "random" tests
#Random.seed!(42)

Dim = 2
degree = 7

# use a unit HyperRectangle 
root = Cell(SVector(ntuple(i -> 0.0, Dim)),
            SVector(ntuple(i -> 1.0, Dim)),
            CellData(Vector{Int}(), Vector{Int}()))

# DGD dof locations
num_basis = binomial(Dim + degree, Dim)

num_nodes = 50*num_basis
points = rand(Dim, num_nodes)

# refine mesh, build sentencil, and evaluate particular quad and nullspace
CutDGD.refine_on_points!(root, points)
for cell in allleaves(root)
    split!(cell, CutDGD.get_data)
end

#points = zeros(Dim, CutDGD.num_leaves(root))
points_init = deepcopy(points)
CutDGD.build_nn_stencils!(root, points, degree)

# get reference distance 
#kdtree = KDTree(points, leafsize = 10)
#dist_ref = zeros(num_nodes)
#for i = 1:num_nodes
#    indices, dists = knn(kdtree, view(points,:,i), 2, true)
#    dist_ref[i] = max(dists[2], 1e-3)
#end
#println(dist_ref)
dist_ref = ones(num_nodes)
mu = 0.0

g = zeros(num_nodes*Dim)
g_pert = zero(g)
#y = zero(g)
p = zero(g) 

# try BFGS before implementing its limited memory version 
# Hess_inv = diagm(ones(num_nodes*Dim))

alpha = 1.0
max_iter = 1000
max_line = 10
max_rank = 20

for d = 1:degree
    println("Starting optimization with degree = ",d)

    H_tol = ones(num_nodes)
    H_tol .*= 1e-5

for n = 1:max_iter
    global points
    #obj = CutDGD.obj_norm(root, wp, Z, y, rho, num_nodes)
    #CutDGD.obj_norm_grad!(g, root, wp, Z, y, rho, num_nodes)
    obj = CutDGD.penalty(root, points, points_init, dist_ref, H_tol, mu, d)
    # y[:] = g[:]
    CutDGD.penalty_grad!(g, root, points, points_init, dist_ref, H_tol, mu, d)
    println("iter ",n,": obj = ",obj,": norm(grad) = ",norm(g))

    # if n > 1
    #     y[:] -= g[:]
    #     y[:] .*= -1.0
    #     rho = 1/dot(y, p)
    #     Hess_inv += (I + rho * p *y')*Hess_inv*(I + rho * y * p')
    #     Hess_inv += rho * p * p'
    # else
    #     Hess_inv ./= norm(g)
    # end

    H = CutDGD.diagonal_norm(root, points, d)
    # compute the discrete KS function's denom 
    minH = minimum(H)
    println("min H = ",minH)
    if minH > 0.0 
        break
    end
        
    if false
        # This "1D Newtons' method" works, but it is slow to converge
        p[:] = -g[:]/norm(g)
        dxc = reshape(p, (Dim, num_nodes))
        eps_fd = 1e-6
        points += eps_fd*dxc
        CutDGD.penalty_grad!(g_pert, root, points, points_init, dist_ref, mu, d)
        pHp = dot(p, (g_pert - g)/eps_fd)
        if pHp > 1e-3
            alpha = -dot(g, p)/(pHp) # negative accounted for below 
        else
            alpha = 1.0
        end
        points -= eps_fd*dxc
    end

    CutDGD.apply_approx_inverse!(p, g, root, points, dist_ref, H_tol, mu, d, 
                                 max_rank)
    alpha = 1.0 
    dxc = reshape(p, (Dim, num_nodes))

    #CutDGD.penalty_block_hess!(p, g, root, points, dist_ref, mu, degree)    
    #global Hess_inv
    #global p = Hess_inv*g

    obj0 = obj
    #dxc = reshape(p, (Dim, num_nodes))
    #alpha = 1.0
    for k = 1:max_line
        points += alpha*dxc
        obj = CutDGD.penalty(root, points, points_init, dist_ref, H_tol, mu, d)
        println("\tline-search iter ",k,": alpha = ",alpha,": obj0 = ",obj0,": obj = ",obj)
        if obj < obj0
            break
        end
        points -= alpha*dxc
        alpha *= 0.1        
    end
    
end
end # degree loop

using PyPlot

PyPlot.plot([0,1,1,0,0],[0,0,1,1,0], "k-")
PyPlot.plot(vec(points_init[1,:]), vec(points_init[2,:]), "ro")
PyPlot.plot(vec(points[1,:]), vec(points[2,:]), "ko")
dx = points - points_init
PyPlot.quiver(vec(points_init[1,:]), vec(points_init[2,:]), vec(dx[1,:]), vec(dx[2,:]), angles="xy", scale_units="xy", scale=1, lw=1)
PyPlot.axis("equal")
#PyPlot.axis([0, 1, 0, 1])

PyPlot.figure()
PyPlot.plot(vec(points_init[1,:]), vec(points_init[2,:]), "ro")
PyPlot.axis("equal")
PyPlot.figure()
PyPlot.plot(vec(points[1,:]), vec(points[2,:]), "ko")
PyPlot.axis("equal")

end # module 