module MomentDebug

using CutDGD
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using LevelSets
using CxxWrap
using SparseArrays
using PyPlot
using CutQuad

# Following StaticArrays approach of using repeatable "random" tests
Random.seed!(42)

plot = false
degree = 6
Dim = 3

# use a unit HyperRectangle 
root = Cell(SVector(ntuple(i -> 0.0, Dim)),
SVector(ntuple(i -> 1.0, Dim)), 
CellData(Vector{Int}(), Vector{Int}()))

if false
    # define a level-set that cuts the HyperRectangle
    num_basis = 1
    xc = 0.5*ones(Dim, num_basis)
    xc[1, 1] = 1/pi
    nrm = zeros(Dim, num_basis)
    nrm[1, 1] = 1.0/sqrt(2)
    nrm[2, 1] = 1.0/sqrt(2)
    tang = zeros(Dim, Dim-1, num_basis)
    tang[:, :, 1] = nullspace(reshape(nrm[:, 1], 1, Dim))
    crv = zeros(Dim-1, num_basis)
else 
    # define a level-set for a circle 
    num_basis = 256
    #xc = zeros(Dim, num_basis)
    xc = randn(Dim, num_basis)
    nrm = zero(xc) 
    tang = zeros(Dim, Dim-1, num_basis)
    crv = zeros(Dim-1, num_basis)
    R = 1/3
    for i = 1:num_basis 
        # theta = 2*pi*(i-1)/(num_basis-1)
        # xc[:,i] = R*[cos(theta); sin(theta)] + [0.5;0.5]
        # nrm[:,i] = [cos(theta); sin(theta)]
        nrm[:,i] = xc[:,i]/norm(xc[:,i])
        xc[:,i] = R*nrm[:,i] + 0.5*ones(Dim) #[0.5;0.5]
        tang[:,:,i] = nullspace(reshape(nrm[:, i], 1, Dim))
        crv[:,i] .= 1/R
    end
end
rho = 100.0*num_basis
levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

# Generate some DGD dof locations used to refine the background mesh
num_nodes = 10*binomial(Dim + degree, Dim)
points = rand(Dim, num_nodes)
CutDGD.refine_on_points!(root, points)
for cell in allleaves(root)
    split!(cell, CutDGD.get_data)
end

CutDGD.mark_cut_cells!(root, levset)


num_cell = CutDGD.num_leaves(root)
cell_xavg = zeros(Dim, num_cell)
cell_dx = ones(Dim, num_cell) #zero(cell_xavg)
for (c,cell) in enumerate(allleaves(root))
    cell_xavg[:,c] = center(cell)
    cell_dx[:,c] = 2*cell.boundary.widths
end

num_basis = binomial(Dim + degree, Dim)
moments = zeros(num_basis, num_cell)

# get arrays/data used for tensor-product quadrature 
x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
num_quad = length(w1d)^Dim             
wq = zeros(num_quad)
xq = zeros(Dim, num_quad)
Vq = zeros(num_quad, num_basis)
workq = zeros((Dim+1)*num_quad)

# set up the level-set function for passing to calc_cut_quad below
const mod_levset = Ref{Any}()
mod_levset[] = levset
safe_clevset = @safe_cfunction( 
    x -> evallevelset(x, mod_levset[]), Cdouble, (Vector{Float64},)
)
#safe_clevset = @safe_cfunction(
#    x -> norm(x - 0.5*ones(Dim)) - R, Cdouble,  (Vector{Float64},)
#)

if plot
fig = figure("quad_points",figsize=(10,10))
# plot the cells
for leaf in allleaves(root)
    v = hcat(collect(vertices(leaf.boundary))...)
    PyPlot.plot(v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], "-k")
end
theta = range(0, stop=2*pi, length=101) #LinRange(0, 2*pi, 100)
PyPlot.plot(R*cos.(theta) .+ 0.5, R*sin.(theta) .+ 0.5, "--r")
#PyPlot.plot([xc[1,1];  ], [0; 2*xc[1,1]], "--r")

PyPlot.plot(xc[1,:], xc[2,:], "go")
end


for (c, cell) in enumerate(allleaves(root))
    xavg = view(cell_xavg, :, c)    
    dx = view(cell_dx, :, c)
    println("xavg = ",xavg, ": dx = ",dx)
    if cell.data.immersed
        println("\timmersed")
        # do not integrate cells that have been confirmed immersed        
        if plot 
            PyPlot.plot([xavg[1]], [xavg[2]], "ro")
        end        
        continue
    elseif CutDGD.is_cut(cell)
        println("\tuse Saye's algorithm")
        # this cell *may* be cut; use Saye's algorithm
        wq_cut, xq_cut, surf_wts, surf_pts = calc_cut_quad(
        cell.boundary, safe_clevset, degree+1, fit_degree=degree)
        println("\tnumber quad points = ",length(wq_cut))
        if plot
            PyPlot.plot(xq_cut[1,:], xq_cut[2,:], "bs", ms=4)
        end
        # consider resizing 1D arrays here, if need larger
        for I in CartesianIndices(xq_cut)
            xq_cut[I] = (xq_cut[I] - xavg[I[1]])/dx[I[1]] - 0.5
        end
        Vq_cut = zeros(length(wq_cut), num_basis)
        workq_cut = zeros((Dim+1)*length(wq_cut))
        CutDGD.poly_basis!(Vq_cut, degree, xq_cut, workq_cut, Val(Dim))
        for i = 1:num_basis
            moments[i, c] = dot(Vq_cut[:,i], wq_cut)
        end
    else
        println("\tnot immeresed, not cut")
        # this cell is not cut; use a tensor-product quadrature to integrate
        # Wait, these are always the same for uncut cells!!!
        # Precompute
        CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
        if plot
            PyPlot.plot(xq[1,:], xq[2,:], "ks", ms=4)
        end
        for I in CartesianIndices(xq)
            xq[I] = (xq[I] - xavg[I[1]])/dx[I[1]] - 0.5
        end
        CutDGD.poly_basis!(Vq, degree, xq, workq, Val(Dim))
        for i = 1:num_basis
            moments[i, c] = dot(Vq[:,i], wq)
        end
    end
    
end 

# check that (scaled) zero-order moments sum to cut-domain volume scaled
# by the constant basis
vol = 0.0
for (c,cell) in enumerate(allleaves(root))
    global vol += moments[1,c]
end
tet_vol = 2^Dim/factorial(Dim)
basis_val = 1/sqrt(tet_vol)
vol /= basis_val

println("vol = ",vol)
println("true vol = ",1 - (4/3)*pi*R^3)

end #module 
