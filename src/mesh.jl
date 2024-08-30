# Structures and methods associated with the `RegionTree` mesh.  Note that 
# `leaf` and `cell` are used interchangably throughout the code.

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
    cut = is_cut(rect, levset)

Returns true if the HyperRectangle `rect` is intersected by the level-set 
`levset`, and returns false otherwise.
"""
function is_cut(rect::HyperRectangle{Dim,T}, levset::Function;
                fit_degree::Int=2) where {Dim, T}
    # This is a bit of a hack; we call Algoim and if there are no quadrature 
    # points, we assume this cell is cut
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
This method, which is identical to the above method, is used when refining 
cells that are cut by a given levelset.
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

"""
    build_nn_stencils!(root, points, degree)

The stencil for each leaf in the tree `root` is determined and the indices of 
the stencil are stored in the leaves.  The stencil is determined using `k` 
neareast neighbors, where `k` is the number of points needed for a 
well-conditioned Vandermonde matrix of total degree `degree`.
"""
function build_nn_stencils!(root, points, degree)
    kdtree = KDTree(points, leafsize = 10)
    Dim = size(points,1)
    max_stencil_iter = 2*degree + 1 #max(degree^2) + 2  # degree + 1
    num_basis = binomial(Dim + degree, Dim)
    sortres = true
    tol = 5.0
    for leaf in allleaves(root)
        if is_immersed(leaf)
            continue 
        end
        xc = center(leaf)
        for k = 1:max_stencil_iter 
            num_nodes = binomial(Dim + degree, Dim) + k
            #num_nodes = binomial(Dim + degree, Dim) + div(degree + 1,2)*Dim + (k-1) #*degree
            indices, dists = knn(kdtree, xc, num_nodes, sortres)
            # build the Vandermonde matrix and check its condition number
            xpts = points[:, indices]
            xref, dx = get_bounding_box(leaf.boundary, xpts)
            dx *= 1.001
            for I in CartesianIndices(xpts)
                xpts[I] = (xpts[I] - xref[I[1]])/dx[I[1]] - 0.5
            end
            workc = zeros((Dim+1)*num_nodes)
            V = zeros(num_nodes, num_basis)
            poly_basis!(V, degree, xpts, workc, Val(Dim))
            #println("iteration ",k,": cond(V) = ",cond(V))
            if cond(V) < tol*10^degree
                # the condition number is acceptable
                leaf.data.points = indices
                break
            end
            if k == max_stencil_iter || num_nodes == size(points,2)
                # condition number is not acceptable, but we accept it
                leaf.data.points = indices
                break
                #error("Failed to find acceptable stencil.")
            end
        end
    end
    return nothing
end

function extend_stencils!(root, points, H, H_tol)
    kdtree = KDTree(points, leafsize = 10)
    Dim = size(points,1)
    sortres = true
    for leaf in allleaves(root)
        if is_immersed(leaf)
            continue 
        end
        Hc = view(H, leaf.data.points)
        Hc_tol = view(H_tol, leaf.data.points)
        if all(Hc .>= Hc_tol)
            continue 
        end
        # if we get here, at least one node's norm violates its constraint
        xc = center(leaf)
        num_nodes = length(leaf.data.points) + 1
        indices, dists = knn(kdtree, xc, num_nodes, sortres)
        leaf.data.points = indices
    end
    return nothing
end

"""
    set_xref_and_dx!(root, points)

For each `leaf` in `root`, finds a reference position and reference scale for
`leaf.data.xref` and `leaf.data.dx`, respectively.  The reference position is,
roughly, the average of the coordinates `points[:,leaf.data.points]` and the 
`leaf`'s boundary.  The reference scale is the extent of the coordinates and 
boundary.
"""
function set_xref_and_dx!(root, points)
    for leaf in allleaves(root)
        xpts = view(points, :, leaf.data.points)
        leaf.data.xref, leaf.data.dx = get_bounding_box(leaf.boundary, xpts)
        # 1.001 prevents numerical issues in the orthogonal polynomials
        leaf.data.dx[:] .*= 1.001
    end
    return nothing
end

"""
    max_stencil = max_leaf_stencil(root)

Returns the largest stencil (i.e. number of points used for DGD basis) over all 
leaves in the mesh defined by `root`.
"""
function max_leaf_stencil(root)
    max_stencil = 0
    for leaf in allleaves(root)
        max_stencil = max(max_stencil, length(leaf.data.points))
    end
    return max_stencil
end