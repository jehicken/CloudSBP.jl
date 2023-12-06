
"""
Data associated with a RegionTree `Cell`.
"""
mutable struct CellData
    points::Vector{Int}
    faces::Vector{Int}
    cut::Bool
    immersed::Bool 
end

function CellData(points::Vector{Int}, faces::Vector{Int})
    return CellData(points, faces, false, false)
end

"""
    cut = is_cut(rect, levset)

Returns true if the HyperRectangle `rect` _may be_ intersected by the level-set 
`levset`, and returns false otherwise.  The determination regarding cut or
not-cut first uses a level-set bound, which is conservative.  If the bound 
indicates `rect` may be cut, the closest point on the level-set from the center 
of `rect` is used to determine if it may be cut.

**Note:** The quadrature algorithm will gracefully handle elements that are 
marked as cut but are actually not cut, so being conservative is fine.
"""
function is_cut(rect::HyperRectangle{Dim,T}, levset::LevelSet{Dim,T}
                ) where {Dim, T}
    xc = rect.origin + 0.5*rect.widths
    ls, bound = boundlevelset(xc, rect.widths, levset)
    if ls*(ls - sign(ls)*bound) > 0
        # the bound has established rect is not cut 
        return false
    else
        # element may be cut; find closest point 
        x = zeros(Dim)
        if findclosest!(x, xc, levset)
            # Newton's method converged
            L = norm(0.5*rect.widths)            
            if norm(x - xc) > L 
                return false 
            end
        end
        return true 
    end 
end

"""
    inside = is_immersed(rect, levset)

Returns true if the **center** of `rect` is inside the level-set `levset`, that 
is, phi(xc) < 0.0.
"""
function is_immersed(rect::HyperRectangle{Dim,T}, levset::LevelSet{Dim,T}
                     ) where {Dim, T}
    xc = rect.origin + 0.5*rect.widths
    return evallevelset(xc, levset) > 0.0 ? true : false
end

"""
    mark_cut_cells!(root, levset)

Identifies cells in the tree `root` that _may be_ cut be the level-set levset.
"""
function mark_cut_cells!(root::Cell{Data, Dim, T, L}, levset::LevelSet{Dim,T}
                         ) where {Data, Dim, T, L}
    for cell in allleaves(root)
        cell.data.cut = is_cut(cell.boundary, levset)
        cell.data.immersed = is_immersed(cell.boundary, levset) 
    end
end

"""
    r = PointRefinery(point_subset)

Used during mesh refinement; see RegionTree documentation for addiation 
information.
"""
struct PointRefinery{T} <: AbstractRefinery
    x_view::T
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
                    deepcopy(cell.data.faces), cell.data.cut,
                    cell.data.immersed)
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
    child_data = refine_data(r, cell, indices)

Returns data for child of `cell` with `indices`.
"""
function refine_data(r::PointRefinery, cell::Cell, indices)
    child_bnd = child_boundary(cell, indices)
    return CellData(get_points_inside(child_bnd, r.x_view,
                    cell.data.points), deepcopy(cell.data.faces), false, false)
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
neareast neighbors, where `k` is the number of points needed for a 
well-conditioned Vandermonde matrix of total degree `degree`.
"""
function build_nn_stencils!(root, points, degree)
    kdtree = KDTree(points, leafsize = 10)
    # find the necessary nearest neighbors
    dim = size(points,1)
    max_stencil_iter = 5
    sortres = true
    for leaf in allleaves(root)
        # the `degree^2` below was found using experiments on uniform grids
        num_nbr = binomial(dim + degree, dim) + 1 #+ degree #+ degree^2 # + degree
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
