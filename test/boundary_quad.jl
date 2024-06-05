module BoundaryQuad

using CutDGD
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using CxxWrap
using SparseArrays
using CutQuad
using PyPlot

function find_closest(x0, levset, levset_grad; max_dx = 1.0)
    x = deepcopy(x0)
    lambda = 0.0
    max_newton = 10
    rel_tol = 1e-8
    grad0 = 0.0
    res0 = 0.0
    for n = 1:max_newton 
        grad_phi = levset_grad(x)
        dLdx = (x-x0) + lambda*grad_phi 
        phi = levset(x) 
        if n == 1
            grad0 = norm(dLdx)
            res0 = abs(phi)
        else
            println("norm(dLdx) = ",norm(dLdx))
            println("phi(x) = ",phi)
            if (norm(dLdx) < grad0*rel_tol && abs(phi) < res0*rel_tol) || abs(phi) <= 1e-14
                return x 
            end
        end
        lambda = (dot(grad_phi, x0 - x) + phi)/dot(grad_phi, grad_phi)
        dx = x0 - x - lambda*grad_phi 
        if norm(dx) > max_dx 
            dx .*= max_dx/norm(dx)
        end
        x += dx 
    end
    error("newton's method failed")
end


Dim = 3
degree = 4

length = 2.0
cell = Cell(SVector(ntuple(i -> -length/2, Dim)),
            SVector(ntuple(i -> length, Dim)),
            CellData(Vector{Int}(), Vector{Int}()))

if Dim == 1
    levset = x -> (x[1] + 0.5)^2 - 0.25
elseif Dim == 2
    # levset = x -> 4*(x[1] + 1.5)^2 + 36*x[2]^2 - 9
    levset = x -> (x[1] + 1.0)^2 + (x[2] + 1.0)^2 - 2.0
    levset_grad = x -> [2.0*(x[1] + 1.0), 2.0*(x[2] + 1.0)]
elseif Dim == 3
    #levset = x -> (x[1] + 0.5)^2 + x[2]^2 + x[3]^2 - 0.25^2
    levset = x -> (x[1] + 1.0)^2 + (x[2] + 1.0)^2 + (x[3] + 1.0)^2 - 3
    levset_grad = x -> [2.0*(x[1] + 1.0), 2.0*(x[2] + 1.0), 2.0*(x[3] + 1.0)]
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
#num_quad = size(surf_pts,2)
num_basis = binomial(Dim + 2*degree, Dim)

# get the RHS 
b = zeros(Dim*num_basis)
work = zeros((Dim+1)*size(surf_pts,2))
V = zeros(size(surf_pts,2), num_basis)
CutDGD.poly_basis!(V, 2*degree, surf_pts, work, Val(Dim))
for q = 1:size(surf_pts,2) 
    for di = 1:Dim 
        b[num_basis*(di-1)+1:num_basis*di] += V[q,:]*surf_wts[di,q] 
    end
end

num_quad = 3*num_basis

new_pts = length*rand(Dim,num_quad) .+ cell.boundary.origin 
for q = 1:num_quad 
    new_pts[:,q] = find_closest(new_pts[:,q], levset, levset_grad,
                                max_dx = norm(cell.boundary.widths))
end

Vq = zeros(num_quad, num_basis)
work = zeros((Dim+1)*num_quad)
CutDGD.poly_basis!(Vq, 2*degree, new_pts, work, Val(Dim))

# construct the linear system 
A = zeros(Dim*num_basis, num_quad)
for q = 1:num_quad 
    # get the normal vector 
    nrm = -levset_grad(new_pts[:,q])
    nrm[:] ./= norm(nrm)
    for di = 1:Dim 
        A[num_basis*(di-1)+1:num_basis*di,q] += Vq[q,:]*nrm[di]
    end
end
alpha = A\b

#alpha = zeros(num_quad)
new_wts = zeros(Dim, num_quad)
for q = 1:num_quad 
    #swts = ntuple(i -> abs(surf_wts[i,q]) <= 0.0 ? 0.0 : 1.0, Dim)
    nrm = -levset_grad(new_pts[:,q])
    nrm[:] ./= norm(nrm)
    #alpha[q] = dot(nrm, surf_wts[:,q])/dot(nrm, nrm.*swts)/Dim
    new_wts[:,q] = nrm*alpha[q]
end

println("Size(A) = ",size(A))
println("rank(A) = ",rank(A))
println("residual norm = ",norm(A*alpha - b))
println("minimum(alpha) = ",minimum(alpha))
println("maximum(alpha) = ",maximum(alpha))

if Dim == 2
    PyPlot.quiver(vec(new_pts[1,1:num_quad]), vec(new_pts[2,1:num_quad]), vec(new_wts[1,1:num_quad]), vec(new_wts[2,1:num_quad]))
elseif Dim == 3 
    fig = PyPlot.figure()
    ax = fig.gca(projection="3d")
    PyPlot.quiver(vec(new_pts[1,:]), vec(new_pts[2,:]), vec(new_pts[3,:]),
                  vec(new_wts[1,:]), vec(new_wts[2,:]), vec(new_wts[3,:]))
end

end