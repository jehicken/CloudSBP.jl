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

# Following StaticArrays approach of using repeatable "random" tests
Random.seed!(42)

Dim = 2
degree = 2


# use a unit HyperRectangle 
root = Cell(SVector(ntuple(i -> 0.0, Dim)),
SVector(ntuple(i -> 1.0, Dim)),
CellData(Vector{Int}(), Vector{Int}()))

levset = x -> norm(x .- SVector(ntuple(i -> 0.5, Dim)))^2 - 0.25^2
#levset = x -> 0.5 - x[1]

num_basis = binomial(Dim + 2*degree-1, Dim)
num_nodes = 5*num_basis
xc = rand(Dim, num_nodes)

#xc[1,:] .*= 0.5
for i in axes(xc,2)
    while levset(view(xc,:,i)) < 0.0
        xc[:,i] = rand(Dim)
    end
end

# refine mesh, build stencil, get face lists
CutDGD.refine_on_points!(root, xc)
for cell in allleaves(root)
    split!(cell, CutDGD.get_data)
end
CutDGD.mark_cut_cells!(root, levset)
CutDGD.build_nn_stencils!(root, xc, 2*degree-1)
CutDGD.set_xref_and_dx!(root, xc)
m = CutDGD.calc_moments!(root, levset, 2*degree-1, min(degree,2))
ifaces = CutDGD.build_faces(root)
bfaces = CutDGD.build_boundary_faces(root)
CutDGD.mark_cut_faces!(ifaces, levset)
CutDGD.mark_cut_faces!(bfaces, levset)

# construct the norm, skew and symmetric operators 
H = zeros(num_nodes)
CutDGD.diagonal_norm!(H, root, xc, 2*degree-1)
S = CutDGD.skew_operator(root, ifaces, bfaces, xc, levset, degree,
fit_degree=min(degree,2))
E = CutDGD.symmetric_operator(root, ifaces, bfaces, xc, levset, degree,
fit_degree=min(degree,2))

# check that H*dV/dx = (S + 0.5*E)*V 
num_basis = binomial(Dim + degree, Dim)
V = zeros(num_nodes, num_basis)
CutDGD.monomial_basis!(V, degree, xc, Val(Dim))
dV = zeros(num_nodes, num_basis, Dim)
CutDGD.monomial_basis_derivatives!(dV, degree, xc, Val(Dim))

# check global compatibility 
for di = 1:Dim
    y = zeros(num_nodes, num_basis)
    rows = rowvals(E[di])
    vals = nonzeros(E[di])
    ~, n = size(E[di])
    for j = 1:n
        for i in nzrange(E[di], j)
            row = rows[i]
            val = vals[i]
            y[row,:] += val*V[j,:]
            if row != j
                y[j,:] += val*V[row,:]
            end
        end
    end
    VtEV = V'*y
    VtHdV = V'*(H.*dV[:,:,di])
    VtHdV += VtHdV'
    #@test isapprox(norm(VtHdV - VtEV), 0.0, atol=10^degree*1e-11)
    println("in test: norm(VtHdV - VtEV)  = ",norm(VtHdV - VtEV))
    
end


fig = figure("quad_points",figsize=(10,10))
num_quad1d = ceil(Int, (2*degree-1+1)/2)
x1d, w1d = CutDGD.lg_nodes(num_quad1d) # could also use lgl_nodes
wq = zeros(length(w1d)^Dim)
xq = zeros(Dim, length(wq))

# plot the cells
for cell in allleaves(root)
    if CutDGD.is_immersed(cell) 
        continue
    end
    v = hcat(collect(vertices(cell.boundary))...)
    PyPlot.plot(v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], "-k")
    if CutDGD.is_cut(cell)
        wq_cut, xq_cut = cut_cell_quad(cell.boundary, levset, num_quad1d, 
                                       fit_degree=min(degree,2))
        PyPlot.plot(vec(xq_cut[1,:]), vec(xq_cut[2,:]), "gs")

        surf_wts, surf_pts = cut_surf_quad(cell.boundary, levset, num_quad1d,
                                           fit_degree=min(degree,2))
        PyPlot.plot(vec(surf_pts[1,:]), vec(surf_pts[2,:]), "rd")
    else
        CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
        PyPlot.plot(vec(xq[1,:]), vec(xq[2,:]), "bs")
    end
end

# plot the interface quad points
for face in ifaces 
    if CutDGD.is_immersed(face)
        continue
    end
    if CutDGD.is_cut(face)
        wq_face, xq_face = cut_face_quad(face.boundary, face.dir, levset,
                                         num_quad1d, fit_degree=min(degree,2))
        PyPlot.plot(vec(xq_face[1,:]), vec(xq_face[2,:]), "rd")
    else
        wq_face = zeros(length(w1d)^(Dim-1))
        xq_face = zeros(Dim, length(wq_face))
        CutDGD.face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)
        PyPlot.plot(vec(xq_face[1,:]), vec(xq_face[2,:]), "bd")
    end

end


# plot the nodes 
PyPlot.plot(vec(xc[1,:]), vec(xc[2,:]), "ko", ms=10, mfc="w")

