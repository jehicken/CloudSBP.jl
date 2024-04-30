module BoundaryQuad

using CutDGD
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using LevelSets
using CxxWrap
using SparseArrays
using CutQuad
using PyPlot

Dim = 2
degree = 16

cell = Cell(SVector(ntuple(i -> -0.5, Dim)),
            SVector(ntuple(i -> 1.0, Dim)),
            CellData(Vector{Int}(), Vector{Int}()))

if Dim == 1
    levset = x -> (x[1] + 0.5)^2 - 0.25
elseif Dim == 2
    # levset = x -> 4*(x[1] + 1.5)^2 + 36*x[2]^2 - 9
    levset = x -> (x[1] + 0.5)^2 + (x[2] + 0.5)^2 - 0.5^2
    levset_grad = x -> [2.0*(x[1] + 0.5), 2.0*(x[2] + 0.5)]
elseif Dim == 3
    #levset = x -> (x[1] + 0.5)^2 + x[2]^2 + x[3]^2 - 0.25^2
    levset = x -> (x[1] + 0.5)^2 + (x[2] + 0.5)^2 + (x[3] + 0.5)^2 - 0.5^2
    levset_grad = x -> [2.0*(x[1] + 0.5), 2.0*(x[2] + 0.5), 2.0*(x[3] + 0.5)]
end

# create a point cloud 
num_basis = binomial(Dim + degree, Dim)
num_nodes = binomial(Dim + 2*degree -1, Dim)
xc = randn(Dim, num_nodes)
cell.data.points = 1:num_nodes
CutDGD.set_xref_and_dx!(cell, xc)
cell.data.cut = true

surf_wts, surf_pts = cut_surf_quad(cell.boundary, levset, degree+1,
                                   fit_degree=min(2,degree))
surf_wts .*= -1.0 

println("size(surf_wts) = ",size(surf_wts))
for di = 1:Dim
    println("surf_wts[",di,",:] = ",surf_wts[di,:])
end

# get the basis functions at surf_pts
num_quad = size(surf_pts,2)
num_basis = binomial(Dim + 2*degree, Dim)
Vq = zeros(num_quad, num_basis)
work = zeros((Dim+1)*num_quad)
CutDGD.poly_basis!(Vq, 2*degree, surf_pts, work, Val(Dim))

# construct the linear system 
A = zeros(Dim*num_basis, num_quad)
b = zeros(Dim*num_basis)
println("Size(A) = ",size(A))
for q = 1:num_quad 
    # get the normal vector 
    nrm = -levset_grad(surf_pts[:,q])
    nrm[:] ./= norm(nrm)
    for di = 1:Dim 
        b[num_basis*(di-1)+1:num_basis*di] += Vq[q,:]*surf_wts[di,q] 
        A[num_basis*(di-1)+1:num_basis*di,q] += Vq[q,:]*nrm[di]
    end
end
#alpha = A\b 

alpha = zeros(num_quad)
new_wts = zero(surf_wts)
for q = 1:num_quad 
    swts = ntuple(i -> abs(surf_wts[i,q]) <= 0.0 ? 0.0 : 1.0, Dim)
    nrm = -levset_grad(surf_pts[:,q])
    nrm[:] ./= norm(nrm)
    alpha[q] = dot(nrm, surf_wts[:,q])/dot(nrm, nrm.*swts)/Dim
    new_wts[:,q] = nrm*alpha[q]
end

println("residual norm = ",norm(A*alpha - b))

if Dim == 2
    PyPlot.quiver(vec(surf_pts[1,:]), vec(surf_pts[2,:]), vec(new_wts[1,:]),
                  vec(new_wts[2,:]))
elseif Dim == 3 
    fig = PyPlot.figure()
    ax = fig.gca(projection="3d")
    PyPlot.quiver(vec(surf_pts[1,:]), vec(surf_pts[2,:]), vec(surf_pts[3,:]),
                  vec(new_wts[1,:]), vec(new_wts[2,:]), vec(new_wts[3,:]))
end

end