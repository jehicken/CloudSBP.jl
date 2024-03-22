
"""
Struct for interfaces and boundary faces of cells.
"""
mutable struct Face{Dim, T, Cell}
    "Face orientation."
    dir::Int
    "Geometric information for the face."
    boundary::HyperRectangle{Dim, T}
    "Adjacent cell(s)."
    cell::SVector{2,Union{Cell, Nothing}}
    "If false, cell is not cut; may or may not be cut otherwise."
    cut::Bool
    "If true, cell is not cut and its center is immersed."
    immersed::Bool 
end

"""
    f = Face(dir, origin, widths [, data=nothing])

Construct a face with normal direction index `dir`, with corner at `origin`, 
and having `widths` dimensions. 
"""
function Face(dir::Int, origin::SVector{Dim, T}, widths::SVector{Dim, T}, 
              left::Union{Cell,Nothing}=nothing,
              right::Union{Cell,Nothing}=nothing) where {Dim, T, Cell}
    @assert( abs(widths[abs(dir)]) < eps(T) )
    Face(dir, HyperRectangle(origin, widths),
         SVector{2,Union{Cell, Nothing}}(left, right),
         false, false)
end

"""
    f = build_face(dir, left, right)

Construct a face with normal direction index `dir` having adjacent cells 
`left` and `right`.  With this constructor, `dir` is always positive, and 
points from `left` to `right`.
"""
function build_face(dir::Int, left::Cell{Data, Dim, T, L},
                    right::Cell{Data, Dim, T, L}) where {Data, Dim, T, L}
    @assert( dir >= 1 && dir <= 3, "dir must be 1, 2, or 3 for interface")
    origin = SVector(
        ntuple(i -> i == dir ? right.boundary.origin[i] : 
               max(left.boundary.origin[i],right.boundary.origin[i]), Dim))
    widths = SVector(
        ntuple(i -> i == dir ? 0.0 : 
               min(left.boundary.widths[i], right.boundary.widths[i]), Dim))
    Face(dir, origin, widths, left, right)
end

"""
    f = build_boundary_face(dir, cell)

Construct a boundary face with normal direction index `dir` having adjacent 
element `cell`.  With this constructor, `dir` can be negative.  For example, 
`dir=1` is an East face, while `dir=-1` is a West face.
"""
function build_boundary_face(dir::Int, cell::Cell{Data, Dim, T, L}
                             ) where {Data, Dim, T, L}
    if dir > 0 
        origin = SVector(
            ntuple(i -> i == dir ? 
                   cell.boundary.origin[i] + cell.boundary.widths[i] :
                   cell.boundary.origin[i], Dim))
    else 
        origin = SVector(cell.boundary.origin)
    end
    widths = SVector(
        ntuple(i -> i == abs(dir) ? 0.0 : cell.boundary.widths[i], Dim))
    Face(dir, origin, widths, cell, nothing)
end

show(io::IO, face::Face) = print(io, "Face: dir = $(face.dir) $(cell.boundary)")


"""
    cut = is_cut(face)

Returns `false` if `face` is not cut; note that a return of `true`, however, 
only indicates that the face *may* be cut.
"""
function is_cut(face::Face{Dim, T, Cell}) where {Dim, T, Cell}
    return face.cut
end

"""
    im = is_immersed(face)

Returns `true` if face is not cut and its center is immersed.
"""
function is_immersed(face::Face{Dim, T, Cell}) where {Dim, T, Cell}
    return face.immersed
end

"""
    I = indices_within_parent(cell)

Returns the Cartesian indices of `cell` with respect to its parent.  This 
function assumes `cell` has a parent and will fail if it does not.
"""
function indices_within_parent(cell)
    p = parent(cell)
    @assert( !(p === nothing) , "cell cannot be root")
    for I in RegionTrees.child_indices(p)
        if cell === p[I...]
            return I
        end
    end
end

"""
    nbr = neighbor_of_greater_or_equal_size(cell, side)

Returns the cell on boundary index `side` of `cell` that is of the same size or 
larger than `cell`.  Returns `nothing`` if no such neighbor exists.
"""
function neighbor_of_greater_or_equal_size(cell::Cell{Data, Dim, T, L},
        side::Int)::Union{Nothing,Cell{Data, Dim, T, L}} where {Data, Dim, T, L}
    p = parent(cell)
    if p === nothing        
        # cell is the root, so there are no neighbors
        return nothing
    end
    dir = div(side + 1, 2)
    I = indices_within_parent(cell)
    if I[dir] == side % 2 + 1 
        # cell's neighbor is contained in parent, and is given by index J
        J = CartesianIndex(
            ntuple(i -> i == dir ? (side + 1) % 2 + 1 : I[i], Dim))
        return p[J]
    end 
    # if we get here, cell was not on the "nice" side of parent p; go up tree 
    pnbr = neighbor_of_greater_or_equal_size(p, side)
    if pnbr === nothing || isleaf(pnbr)
        return pnbr
    end
    # if we get here, pnbr is not a leaf, so we need to find which of its 
    # children is the neighbor of cell; however, at this point, we know cell's 
    # position (high or low in direction dir) within p
    J = CartesianIndex(ntuple(i -> i == dir ? side % 2 + 1 : I[i], Dim))
    return pnbr[J]
end

"""
    leaves = leaves_on_side(cell, side)

Returns a Vector of leaves on boundary index `side` of given root `cell`.  If 
`cell` is nothing, the Vector is returned empty.
"""
function leaves_on_side(cell::Cell{Data,Dim,T,L},
                        side::Int) where {Data, Dim, T, L}
    # initialize the list of potential candidate nodes
    candidates = Vector{Cell{Data,Dim,T,L}}()
    if !(cell === nothing)
        push!(candidates, cell)
    end
    leaves = Vector{Cell{Data, Dim, T, L}}()
    dir = div(side + 1, 2)
    while length(candidates) > 0
        node = popfirst!(candidates)
        if isleaf(node)
            # node itself is a leaf, so add it to the Vector
            push!(leaves, node)
        else
            # node is a parent, so add all appropriate children to
            # candidate list 
            for I in RegionTrees.child_indices(node)
                if I[dir] == (side + 1) % 2 + 1
                    push!(candidates, node[I...])
                end 
            end
        end
    end
    return leaves           
end 

"""
    neighbors = get_neighbors(cell, side)

Constructs a vector of all neighbors on the `side` boundary of leaf `cell`.

Based on the paper here: http://www.cs.umd.edu/~hjs/pubs/SameCVGIP89.pdf
See also https://geidav.wordpress.com/2017/12/02/advanced-octrees-4-finding-neighbor-nodes/
"""
function get_neighbors(cell::Cell, side)::Vector{Cell}
    nbr = neighbor_of_greater_or_equal_size(cell, side)
    if nbr === nothing 
        return Vector{Cell}()
    end
    opposite_side = div(side - 1, 2)*2 + side % 2 + 1
    neighbors = leaves_on_side(nbr, opposite_side)
    return neighbors
end

"""
    faces = build_faces(root)

Creates an array of faces between adjacent leaves (i.e. cells) of the tree 
`root`.  This array does not include boundary faces that have only one adjacent 
cell; see `build_boundary_faces` for such a list.
"""
function build_faces(root::Cell{Data, Dim, T, L}) where {Data, Dim, T, L}
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
            end 
            if abs(leaf.boundary.origin[d] + leaf.boundary.widths[d] -
                   root.boundary.origin[d] - root.boundary.widths[d]) < 1e-14
                # This leaf is on side 2*d, so add the appropriate face
                face = build_boundary_face(d, leaf)
                push!(face_list, face)
            end
        end
    end
    return face_list
end

"""
    mark_cut_faces!(faces, levset)

Identifies faces in the list `faces` that _may be_ cut be the level-set levset.
"""
function mark_cut_faces!(faces, levset::LevelSet{Dim,T}) where {Dim, T}
    for face in faces
        face.cut = is_cut(face.boundary, levset)
        if !face.cut && is_center_immersed(face.boundary, levset)
            # This face is definitely not cut, and its center is immersed, so 
            # entire face must be immersed.
            face.immersed = true
        end
    end
end
