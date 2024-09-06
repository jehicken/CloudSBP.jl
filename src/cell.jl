# Structures and methods that build upon the RegionTree Cell.
# Note that `leaf` and `cell` are used interchangably throughout the code.

"""
Data associated with a RegionTree `Cell`.
"""
mutable struct CellData
    "Indices of points in the stencil of the cell."
    points::Vector{Int}
    "Indices of interfaces of the cell."
    faces::Vector{Int}
    "Indices of boundary faces of the cells."
    bfaces::Vector{Int}
    "If false, cell is not cut; may or may not be cut otherwise."
    cut::Bool
    "If true, cell is not cut and its center is immersed."
    immersed::Bool 
    "Integral moments (view, not owned)"
    moments::AbstractVector{Float64}
    "Reference position for conditioning Vandermonde matrices"
    xref::Vector{Float64} # use static array with type param?
    "Reference scaling for conditioning Vandermonde matrices"
    dx::Vector{Float64} # use static array with type param?
    "Cell-based quadrature rule"
    wts::Vector{Float64}
end

function CellData(points::Vector{Int}, faces::Vector{Int}, bfaces::Vector{Int})
    return CellData(points, faces, bfaces, false, false, Float64[], Float64[],
                    Float64[], Float64[])
end

function CellData(points::Vector{Int}, faces::Vector{Int})
    return CellData(points, faces, Vector{Int}(), false, false, Float64[],
                    Float64[], Float64[], Float64[])
end

"""
    box_center, box_dx = get_bounding_box(rect, x)

For a given HyperRectangle `rect` and point cloud `x`, returns the bounding box
center and lengths that enclose both `rect` and `x`.
"""
function get_bounding_box(rect, x)
    lower = vec(minimum([x  rect.origin], dims=2))
    upper = vec(maximum([x (rect.origin + rect.widths)], dims=2))
    return 0.5*(upper + lower), (upper - lower)
end

"""
    cut = is_cut(rect, levset [, fit_degree=2])

Returns true if the HyperRectangle `rect` is intersected by the level-set 
`levset`, and returns false otherwise.
"""
function is_cut(rect::HyperRectangle{Dim,T}, levset::Function;
                fit_degree::Int=2) where {Dim, T}
    # if the level-set is two-times larger than the radius of the hypersphere 
    # enclosing `rect`, we assume that the cell is not cut and not immersed
    xc = rect.origin + 0.5*rect.widths
    if levset(xc) > norm(rect.widths)
        return false
    end
    # If we get here, the cell may be fully immersed or it may be cut...and it # may even be neither of these.
    # This is a bit of a hack; we call Algoim and if there are no quadrature 
    # points, we assume this cell is immersed, otherwise we assume it is cut.
    wts, pts = cut_surf_quad(rect, levset, 1, fit_degree=fit_degree)
    if isempty(wts)
        return false
    else
        return true
    end
end

"""
    cut = is_cut(cell)

Returns `false` if `cell` is not cut; note that a return of `true`, however, 
only indicates that the cell *may* be cut.
"""
function is_cut(cell) 
    return cell.data.cut
end

"""
    im = is_immersed(cell)

Returns `true` if cell is not cut and its center is immersed.
"""
function is_immersed(cell)
    return cell.data.immersed
end

"""
    inside = is_center_immersed(rect, levset)

Returns true if the **center** of `rect` is inside the level-set `levset`, that 
is, phi(xc) < 0.0.
"""
function is_center_immersed(rect::HyperRectangle{Dim,T}, levset::Function
                            ) where {Dim, T}
    xc = rect.origin + 0.5*rect.widths
    return levset(xc) < 0.0 ? true : false
end

"""
    mark_cut_cells!(root, levset)

Identifies cells in the tree `root` that _may be_ cut be the level-set `levset`.
"""
function mark_cut_cells!(root::Cell{Data, Dim, T, L}, levset::Function
                         ) where {Data, Dim, T, L}
    for cell in allleaves(root)
        cell.data.cut = is_cut(cell.boundary, levset)
        if !cell.data.cut && is_center_immersed(cell.boundary, levset)
            # This cell is definitely not cut, and its center is immersed, so 
            # entire cell must be immersed.
            cell.data.immersed = true 
        end
    end
end

"""
    r = PointRefinery(point_subset)

Used during mesh refinement with respect to a given point cloud; see RegionTree 
documentation for addition information.
"""
struct PointRefinery{T} <: AbstractRefinery
    x_view::T
end

"""
    r = LevelSetRefinery(point_subset)

Used during to refine the mesh around a given level-set.  Refines until the cut 
cells are less than `r.min_widths` dimensions.
"""
struct LevelSetRefinery{T} <: AbstractRefinery
    x_view::AbstractMatrix{T}
    levset::Function
    min_widths::Vector{T}
end

"""
    n = num_leaves(root)

Returns the total number of leaves in the given `root`.
"""
function num_leaves(root)
    num_cells = 0
    for cell in allleaves(root)
        num_cells += 1
    end
    return num_cells 
end

"""
    data = get_data(cell, child_indices)

Returns a `CellData` struct based on the given `cell`.
"""
function get_data(cell, child_indices)
    return CellData(deepcopy(cell.data.points), 
                    deepcopy(cell.data.faces), deepcopy(cell.data.bfaces),
                    cell.data.cut, cell.data.immersed, Float64[], Float64[], 
                    Float64[], Float64[])
end

"""
    inside = get_points_inside(rect, points [, indices=[1:size(points,2)]])

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
    return length(cell.data.points) > 1
end

"""
This method is for determining if cut cells need further refinement.
"""
function needs_refinement(r::LevelSetRefinery, cell)
    if !is_cut(cell.boundary, r.levset) 
        return false
    end
    # if we get here, cell is cut; check its size
    for (w, w_min) in zip(cell.boundary.widths, r.min_widths)
        if w > w_min
            return true 
        end
    end
    return false 
end

"""
    child_data = refine_data(r, cell, indices)

Returns data for child of `cell` with `indices`.
"""
function refine_data(r::PointRefinery, cell::Cell, indices)
    child_bnd = child_boundary(cell, indices)
    return CellData(get_points_inside(child_bnd, r.x_view,
                    cell.data.points), deepcopy(cell.data.faces),
                    deepcopy(cell.data.bfaces))
end

"""
This method, which is almost identical to the above method, is used when 
refining cells that are cut by a given levelset.
"""
function refine_data(r::LevelSetRefinery, cell::Cell, indices)
    child_bnd = child_boundary(cell, indices)
    return CellData(get_points_inside(child_bnd, r.x_view,
                    cell.data.points), deepcopy(cell.data.faces),
                    deepcopy(cell.data.bfaces))
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
    return nothing
end

"""
    refine_on_levelset!(root, points, levset, min_widths)

Refines the leaves in `root` until each cut leaf is less than `min_widths`
dimensions.  The array of `points` are needed so that refined leaves have the 
necessary data.
"""
function refine_on_levelset!(root, points, levset, min_widths)
    r = LevelSetRefinery(view(points, :, :), levset, min_widths)
    for leaf in allleaves(root)
        adaptivesampling!(leaf, r)
    end
    return nothing
end