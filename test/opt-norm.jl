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
degree = 2

# use a unit HyperRectangle 
root = Cell(SVector(ntuple(i -> 0.0, Dim)),
            SVector(ntuple(i -> 1.0, Dim)),
            CellData(Vector{Int}(), Vector{Int}()))

# DGD dof locations
num_basis = binomial(Dim + degree, Dim)

num_nodes = 100*num_basis
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
mu = 0.0001

g = zeros(num_nodes*Dim)

alpha = 1.0
max_iter = 1000
max_line = 10
for n = 1:max_iter

    #obj = CutDGD.obj_norm(root, wp, Z, y, rho, num_nodes)
    #CutDGD.obj_norm_grad!(g, root, wp, Z, y, rho, num_nodes)
    obj = CutDGD.penalty(root, points, points_init, dist_ref, mu, degree)
    CutDGD.penalty_grad!(g, root, points, points_init, dist_ref, mu, degree)
    println("iter ",n,": obj = ",obj,": norm(grad) = ",norm(g))

    H = CutDGD.diagonal_norm(root, points, degree)
    # compute the discrete KS function's denom 
    minH = minimum(H)
    println("min H = ",minH)
    if minH > 0.0 
        break
    end
    
    obj0 = obj
    dxc = reshape(g, (Dim, num_nodes))
    alpha = 10.0
    for k = 1:max_line
        global points -= alpha*dxc
        obj = CutDGD.penalty(root, points, points_init, dist_ref, mu, degree)
        println("\tline-search iter ",k,": obj0 = ",obj0,": obj = ",obj)
        if obj < obj0
            break
        end
        points += alpha*dxc
        alpha *= 0.1        
    end
    
end

using PyPlot

PyPlot.plot([0,1,1,0,0],[0,0,1,1,0], "k-")
PyPlot.plot(vec(points_init[1,:]), vec(points_init[2,:]), "ro")
PyPlot.plot(vec(points[1,:]), vec(points[2,:]), "ko")
dx = points - points_init
PyPlot.quiver(vec(points_init[1,:]), vec(points_init[2,:]), vec(dx[1,:]), vec(dx[2,:]), angles="xy", scale_units="xy", scale=1, lw=1)
PyPlot.axis("equal")
#PyPlot.axis([0, 1, 0, 1])

end # module 