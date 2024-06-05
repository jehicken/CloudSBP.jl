module OptNorm

using CutDGD
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using SparseArrays
using NearestNeighbors

# Following StaticArrays approach of using repeatable "random" tests
#Random.seed!(42)

Dim = 2
degree = 3

# use a unit HyperRectangle 
root = Cell(SVector(ntuple(i -> 0.0, Dim)),
            SVector(ntuple(i -> 1.0, Dim)),
            CellData(Vector{Int}(), Vector{Int}()))

# define a level-set for a circle 
num_basis = 8*4^(Dim-1)
xc = randn(Dim, num_basis)
nrm = zero(xc) 
tang = zeros(Dim, Dim-1, num_basis)
crv = zeros(Dim-1, num_basis)
R = 1/4
for i = 1:num_basis 
    #theta = 2*pi*(i-1)/(num_basis-1)
    #xc[:,i] = R*[cos(theta); sin(theta)] + [0.5;0.5]
    #nrm[:,i] = [cos(theta); sin(theta)]
    nrm[:,i] = xc[:,i]/norm(xc[:,i])
    xc[:,i] = nrm[:,i]*R + 0.5*ones(Dim)
    tang[:,:,i] = nullspace(reshape(nrm[:, i], 1, Dim))
    crv[:,i] .= 1/R
end
rho = 100.0*(num_basis)^(1/(Dim-1))
levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)
levset_func(x) = evallevelset(x, levset)
            
# DGD dof locations
num_basis = binomial(Dim + degree, Dim)

num_nodes = 50*num_basis
points = rand(Dim, num_nodes)
for i = 1:num_nodes
    while norm(points[:,i] - 0.5*ones(Dim)) < R
        points[:,i] = rand(Dim)
    end
end

if false
# make points cluster at left end
num1d = 40
num_nodes = num1d^2
points = zeros(Dim, num_nodes)
ptr = 1
dx = 1/(num1d-1)
for i = 1:num1d
    x = 1 - cos(0.5*pi*(i-1)*dx)
    for j = 1:num1d 
        y = (j-1)*dx
        points[:,ptr] = [x,y] + 0.2*randn(2).*[0.5*pi*dx*sin(0.5*pi*(i-1)*dx); dx]
        global ptr += 1
    end
end
end 

# refine mesh, build sentencil, and evaluate particular quad and nullspace
CutDGD.refine_on_points!(root, points)
for cell in allleaves(root)
    split!(cell, CutDGD.get_data)
end
CutDGD.mark_cut_cells!(root, levset)
CutDGD.build_nn_stencils!(root, points, degree)
CutDGD.set_xref_and_dx!(root, points)
m = CutDGD.calc_moments!(root, levset_func, degree)
vol = sum(m[1,:])
tet_vol = 2^Dim/factorial(Dim)
vol *= sqrt(tet_vol)
println("vol = ",vol)


# get reference distance 
#kdtree = KDTree(points, leafsize = 10)
#dist_ref = zeros(num_nodes)
#for i = 1:num_nodes
#    indices, dists = knn(kdtree, view(points,:,i), 2, true)
#    dist_ref[i] = max(dists[2], 1e-3)
#end
#println(dist_ref)
dist_ref = ones(num_nodes)
mu = 0.1

max_rank = 50 # was 50
H_tol = ones(num_nodes)
H_tol .*= 0.1*vol/num_nodes #  0.5e-5
#kdtree = KDTree(points_init, leafsize = 10)
#min_dist = 1e100
#for i = 1:num_nodes
#    idxs, dists = knn(kdtree, points_init[:,i], 2, true)
#    global min_dist = min(min_dist, dists[2])
#end
#H_tol .*= (min_dist)^Dim
#println("H_tol = ", min_dist^Dim)

points_init = copy(points)
H = CutDGD.opt_norm!(root, points, degree, H_tol, mu, dist_ref, max_rank)

println("minimum(H) = ",minimum(H))

# How many local weights are negative?
count = 0
for cell in allleaves(root)
    if CutDGD.is_immersed(cell)
        continue
    end
    # get the nodes in this cell's stencil
    nodes = view(points, :, cell.data.points)
    # get cell quadrature and add to global norm
    w = CutDGD.cell_quadrature(degree, nodes, cell.data.moments, cell.data.xref,
                        cell.data.dx, Val(Dim))
    if minimum(w) < 0.0
        global count += 1
    end
end
println("Number of cells with negative norm: ",count)
println("Percentage of cells: ",100*count/CutDGD.num_leaves(root)," %")



using PyPlot

theta = LinRange(0, 2*pi, 100)
PyPlot.plot([0,1,1,0,0],[0,0,1,1,0], "k-")
PyPlot.plot(vec(points_init[1,:]), vec(points_init[2,:]), "ro")
PyPlot.plot(vec(points[1,:]), vec(points[2,:]), "ko")
dx = points - points_init
PyPlot.quiver(vec(points_init[1,:]), vec(points_init[2,:]), vec(dx[1,:]), vec(dx[2,:]), angles="xy", scale_units="xy", scale=1, lw=1)
PyPlot.plot(R*cos.(theta) .+ 0.5, R*sin.(theta) .+ 0.5, "-b")
PyPlot.axis("equal")
#PyPlot.axis([0, 1, 0, 1])

PyPlot.figure()
PyPlot.plot(vec(points_init[1,:]), vec(points_init[2,:]), "ro")
PyPlot.plot(R*cos.(theta) .+ 0.5, R*sin.(theta) .+ 0.5, "-b")
PyPlot.axis("equal")
PyPlot.figure()
PyPlot.plot(vec(points[1,:]), vec(points[2,:]), "ko")
PyPlot.plot(R*cos.(theta) .+ 0.5, R*sin.(theta) .+ 0.5, "-b")
PyPlot.axis("equal")

end # module 