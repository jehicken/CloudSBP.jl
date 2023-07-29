module CutDGD

using LinearAlgebra
using SparseArrays
using RegionTrees
import RegionTrees: AbstractRefinery, needs_refinement, refine_data
using StaticArrays: SVector, @SVector
using NearestNeighbors
using CutQuad
import SpecialFunctions: gamma

export CellData, Face

"""
    get_complex_step(T)
returns an appropriate complex-step size for the given type
"""
get_complex_step(::Type{T}) where {T <: Float32} = 1f-20
get_complex_step(::Type{T}) where {T <: Float64} = 1e-60
get_complex_step(::Type{T}) where {T <: ComplexF32} = 1f-20
get_complex_step(::Type{T}) where {T <: ComplexF64} = 1e-60


include("orthopoly.jl")
include("face.jl")
include("dgd_basis.jl")
include("mass.jl")
include("first_derivative.jl")

# The following needs a home in a separate file 

mutable struct CellData
    points::Vector{Int}
    faces::Vector{Int} 
end

struct PointRefinery{T} <: AbstractRefinery
    x_view::T
end

function num_leaves(root)
    num_cells = 0
    for cell in allleaves(root)
        num_cells += 1
    end
    return num_cells 
end


function get_data(cell, child_indices)
    return CellData(deepcopy(cell.data.points), 
                    deepcopy(cell.data.faces))
end

"""
    inside = get_points_inside(rect, points [, indices=[:]])

Returns the subset of integers from `indices` corresponding to points from 
`points` that are inside the hyperrectangle `rect`.
"""
function get_points_inside(rect, points,
                           indices::Vector{Int}=Vector{Int}(1:size(points,2)))
    dim = size(points,1)
    inside = Vector{Int}()
    for i in indices
        # check if points[:,i] is inside rectangle
        add = true
        for d = 1:dim
            if (rect.origin[d] > points[d,i] || 
                rect.origin[d] + rect.widths[d] < points[d,i])
                # this point is outside of this cell
                add = false
            end
        end
        add ? append!(inside, i) : nothing
    end
    return inside
end

"""
    refine = needs_refinement(r, cell)

Returns true if `cell` has more than one point in it.
"""
function needs_refinement(r::PointRefinery, cell)
    #println(cell)
    #println("c.data.point_count = ",cell.data.point_count)
    #return cell.data.point_count > 1
    return length(cell.data.points) > 1
end

"""
    child_data = refine_data(r, cell, indices)

Returns data for child of `cell` with `indices`.
"""
function refine_data(r::PointRefinery, cell::Cell, indices)
    child_bnd = child_boundary(cell, indices)
    return CellData(get_points_inside(child_bnd, r.x_view,
                    cell.data.points), deepcopy(cell.data.faces))
end

"""
    refine_on_points!(root, points)

Refines the tree `root` until each cell has at most one of the points in 
`points`. 
"""
function refine_on_points!(root, points)
    root.data.points = get_points_inside(root.boundary, points)
    r = PointRefinery(view(points, :, :))
    adaptivesampling!(root, r)
end

"""
    build_nn_stencils!(root, points, degree)

The stencil for each leaf in the tree `root` is determined and the indices of 
the stencil are stored in the leaves.  The stencil is determined using `k` 
neareast neighbors, where `k` is the number of points needed for a polynomial 
basis of total degree `degree`.
"""
function build_nn_stencils!(root, points, degree)
    kdtree = KDTree(points, leafsize = 10)
    # find the necessary nearest neighbors
    dim = size(points,1)
    max_stencil_iter = 5
    sortres = true
    for leaf in allleaves(root)
        # the `degree^2` below was found using experiments on uniform grids
        num_nbr = binomial(dim + degree, dim) + degree^2 # + degree # + degree^2
        #num_nbr = (degree+1)^dim
        xc = center(leaf)
        indices, dists = knn(kdtree, xc, num_nbr, sortres)
        #println(indices)
        #println(dists)
        #V = calcVandermonde(xc, view(points, :, indices), degree) 
        leaf.data.points = indices
        # for k = 1:max_stencil_iter
        #     indices, dists = knn(tree, points, num_nbr, sortres = true)
        #     # build the Vandermonde matrix and check its condition number
        #     V = calcVandermonde(xc, view(points, :, indices), degree)
        #     if cond(V) < tol*exp(degree)
        #         break
        #     end 
        # end
    end
end


end
