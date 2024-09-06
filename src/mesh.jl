# Structures and methods for building the background mesh

"""
Mesh data structure
"""
mutable struct Mesh{Dim, T, CellType}
    "Root cell of a RegionTree."
    root::CellType
    "Interfaces between adjacent cells."
    ifaces::Vector{Face{Dim, T, CellType}}
    "Boundary faces on the boundary of the root cell."
    bfaces::Vector{Face{Dim, T, CellType}}
end

"""
    build_nn_stencils!(root, points, degree [, tol=5*10^degree, 
                       max_iter=2*degree+1])

The stencil for each leaf in the tree `root` is determined and the indices of 
the stencil are stored in the leaves.  The stencil is determined using `k` 
neareast neighbors, where `k` is the number of points needed for a 
well-conditioned Vandermonde matrix of total degree `degree`.  The tolerance 
`tol` is used to determine what is considered well-conditioned.  A maximum of 
`max_iter` iterations are used to find a suitable stencil; after the maximum 
iterations are exceeded, the method accepts the stencil as is.
"""
function build_nn_stencils!(root, points, degree; tol::Float64=5.0*10^degree,
                            max_iter::Int=2*degree + 1)
    kdtree = KDTree(points, leafsize = 10)
    Dim = size(points,1)
    num_basis = binomial(Dim + degree, Dim)
    sortres = true
    for leaf in allleaves(root)
        if is_immersed(leaf)
            continue 
        end
        xc = center(leaf)
        for k = 1:max_iter 
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
            if cond(V) < tol
                # the condition number is acceptable
                leaf.data.points = indices
                break
            end
            if k == max_iter || num_nodes == size(points,2)
                # condition number does not meet the tolerance, but we accept it
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

"""
    faces = build_interfaces(root)

Creates an array of faces between adjacent leaves (i.e. cells) of the tree 
`root`.  This array does not include boundary faces that have only one adjacent 
cell; see `build_boundary_faces` for such a list.
"""
function build_interfaces(root::Cell{Data, Dim, T, L}) where {Data, Dim, T, L}
    face_list = Vector{Face{Dim, T, Cell{Data, Dim, T, L}}}()
    for leaf in allleaves(root)
        for d = 1:Dim
            neighbors = get_neighbors(leaf, d*2-1)
            #println("length(neighbors) = ",length(neighbors))
            for nbr in neighbors
                face = build_face(d, nbr, leaf)
                push!(face_list, face)
                # add face to leaf and nbr's lists
                face_index = length(face_list)
                push!(leaf.data.faces, face_index)
                push!(nbr.data.faces, face_index)
            end
        end
    end
    return face_list
end

"""
    faces = build_boundary_faces(root)

Creates an array of boundary faces of the tree `root`.  Boundary faces are 
faces of the leaves of `root` that touch the the East, West, North, ...etc 
sides of `root`.
"""
function build_boundary_faces(root::Cell{Data, Dim, T, L}) where {Data,Dim,T,L}
    face_list = Vector{Face{Dim, T, Cell{Data, Dim, T, L}}}()
    for leaf in allleaves(root)
        for d = 1:Dim 
            if abs(leaf.boundary.origin[d] - root.boundary.origin[d]) < 1e-14
                # This leaf is on side 2*d - 1, so add a face 
                face = build_boundary_face(-d, leaf)
                push!(face_list, face)
                face_index = length(face_list)
                push!(leaf.data.bfaces, face_index)
            end 
            if abs(leaf.boundary.origin[d] + leaf.boundary.widths[d] -
                   root.boundary.origin[d] - root.boundary.widths[d]) < 1e-14
                # This leaf is on side 2*d, so add the appropriate face
                face = build_boundary_face(d, leaf)
                push!(face_list, face)
                face_index = length(face_list)
                push!(leaf.data.bfaces, face_index)
            end
        end
    end
    return face_list
end

"""
    mark_cut_faces!(faces, levset)

Identifies faces in the list `faces` that _may be_ cut be the level-set levset.
"""
function mark_cut_faces!(faces, levset)
    for face in faces
        face.cut = is_cut(face.boundary, levset)
        if !face.cut && is_center_immersed(face.boundary, levset)
            # This face is definitely not cut, and its center is immersed, so 
            # entire face must be immersed.
            face.immersed = true
        end
    end
end

"""
    count = number_immersed(faces)

Returns the number of `faces` that are definitely immersed.  Note that this is 
a lower bound, since the immeresed check is conservative.
"""
function number_immersed(faces)
    count = 0
    for face in faces
        if is_immersed(face)
            count += 1
        end
    end
    return count
end

"""
    mesh = build_mesh(points, widths, levset, min_width 
                      [, origin=SVector(ntuple(i -> 0.0, Dim))])

Builds the background Cartesian mesh.  The root cell has its origin at `origin` 
and has `widths` lengths.  The level-set `levset` defines any immersed boundary 
based on where `levset(x) = 0`; the domain is the intersection of the root 
cell's domain and where `levset(x) >= 0`.  The mesh is refined based on the 
given points `points` and it is refined at the zero level-set until the cut 
cell have dimensions of `min_width` or smaller.

**NOTE**: If you want a degree `p` SBP operator, set `degree=2*p-1` so that the 
stencil is sufficiently large for the diagonal mass matrix.
"""
function build_mesh(points, widths::SVector{Dim,T}, levset, min_widths;
                    origin::SVector{Dim,T}=SVector(ntuple(i -> 0.0, Dim))
                    ) where {Dim, T}

    # generate the root cell 
    root = Cell(origin, widths, CellData(Vector{Int}(), Vector{Int}()))
    
    # refine mesh on points and levelset 
    refine_on_points!(root, points)
    refine_on_levelset!(root, points, levset, min_widths)
    mark_cut_cells!(root, levset)

    # build face lists
    ifaces = build_interfaces(root)
    bfaces = build_boundary_faces(root)
    mark_cut_faces!(ifaces, levset)
    mark_cut_faces!(bfaces, levset)

    # construct and return the mesh 
    return Mesh(root, ifaces, bfaces)
end

"""
    build_cell_stencils!(mesh, points, degree [, tol=5*10^degree, 
                         max_iter=2*degree+1])

This method loops over the cells in `mesh` and constructs the stencil for 
each.  The stencil is based on the nodes `points` and the polynomial `degree`.  
The function also determines the cell reference dimensions for affine scaling 
of the Vandermonde matrix.  See `build_nn_stencils!` for an explanation of 
`tol` and `max_iter`.

**NOTE**: If you want a degree `p` SBP operator, set `degree=2*p-1` so that the 
stencil is sufficiently large for the diagonal mass matrix.

**TODO**: At present this is just a front-end for `build_nn_stencils!`.  In the future we may want an input that allows for different stencil constructions.
"""
function build_cell_stencils!(mesh, points, degree; tol::Float64=5.0*10^degree,
                              max_iter::Int=2*degree + 1)
    # clear any previous stencils
    for cell in allleaves(mesh.root)
        #empty!(cell.data.moments) # moments is just a view
        empty!(cell.data.points)
        empty!(cell.data.dx)
        empty!(cell.data.xref)
        empty!(cell.data.wts)
    end
    
    # build stencil and define reference dimensions for affine scaling
    build_nn_stencils!(mesh.root, points, degree, tol=tol, max_iter=max_iter)
    set_xref_and_dx!(mesh.root, points)
    return nothing
end

"""
    max_stencil, avg_stencil = stencil_stats(mesh)

Returns the maximum cell stencil size and the average stencil size.  Only cells
that are non immersed are included, since immersed cells should have empty 
stencils.

**PRE**: The cells in `mesh` must have their stencils defined; this should be 
the case if a high-level mesh construction was used (e.g. `build_mesh`).
"""
function stencil_stats(mesh::Mesh)
    max_stencil = 0
    avg_stencil = 0
    count = 0
    for cell in allleaves(mesh.root)
        if is_immersed(cell)
            continue
        end
        count += 1
        max_stencil = max(max_stencil, length(cell.data.points))
        avg_stencil += length(cell.data.points)
    end
    avg_stencil = avg_stencil/count
    return max_stencil, avg_stencil
end