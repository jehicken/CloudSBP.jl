
"""
Data for integrals on boundaries

Each boundary condition can be associated with a `BoundaryOperator`, which 
holds data needed to compute integrals over the relevant subset of the 
boundary.  This data is grouped by faces, since each face may require a 
different number of quadrature points or have a different sized stencil.  If 
`f` denotes the index of some face, then `xq_face[f][:,q]` is the `q`th 
quadrature point on face; `nrm_face[f][:,q]` is the quadrature-weighted outward 
facing normal at `xq_face[f][:,q]`; `dof_face[f][i]` is global DOF index 
associated with the local interpolation index `i`, and; `prj_face[f][q,:]` is 
the interpolation operator from the local degrees of freedom to quadrature node 
`q`.
"""
mutable struct BoundaryOperator{T}
    xq_face::Array{Matrix{T}}
    nrm_face::Array{Matrix{T}}
    dof_face::Array{Vector{Int}}
    prj_face::Array{Matrix{T}}
end

"""
    E = BoundaryOperator(T)

Returns a `BoundaryOperator` with empty arrays.
"""
function BoundaryOperator(T::Type)
    BoundaryOperator(Array{Matrix{T}}(undef,0), Array{Matrix{T}}(undef,0),
                     Array{Vector{Int}}(undef,0), Array{Matrix{T}}(undef,0))
end

"""
    xq, nrm, dof, prj = push_new_face!(bndry, Dim, num_nodes, num_quad)

Creates new memory at the end of the fields in the given `BoundaryOperator` and 
returns references to the newly created arrays.
"""
function push_new_face!(bndry::BoundaryOperator{T}, Dim, num_nodes, num_quad
                        ) where {T}
    push!(bndry.xq_face, zeros(Dim, num_quad))
    push!(bndry.nrm_face, zeros(Dim, num_quad))
    push!(bndry.dof_face, zeros(Int, num_nodes))
    push!(bndry.prj_face, zeros(num_quad, num_nodes))
    return bndry.xq_face[end], bndry.nrm_face[end], bndry.dof_face[end], 
        bndry.prj_face[end]
end


"""
Summation-by-parts first derivative operator

`S[d]` holds the skew-symmetric part for direction `d` and `E[:]` holds `BoundaryOperator`s that define the symmetric part.
"""
mutable struct SBP{T,Dim} #FirstDeriv{T, Dim}
    S::SVector{Dim, SparseMatrixCSC{T, Int64}}    
    #E::Matrix{SparseMatrixCSC{T, Int64}}
    bnd_pts::Array{Matrix{T}}
    bnd_nrm::Array{Matrix{T}}
    bnd_dof::Array{Vector{Int}}
    bnd_prj::Array{Matrix{T}}
end


"""
    rect = build_cell_face(dir, cell)

Construct a `HyperRectangle` for given cell on side `dir`.
"""
function cell_side_rect(dir, cell::Cell{Data, Dim, T, L}
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
    return HyperRectangle(origin, widths)
end

"""
    E = cell_symmetric_part(cell, xc, degree)

Returns the symmetric part of the first-derivative SBP operator for the uncut 
cell `cell`.  The point cloud associated with `cell` is `xc`, and the boundary 
operator is `2*degree` exact for boundary integrals.

**Note**: This version recomputes the 1D quadrature rule each time, and involves
several allocations.
"""
function cell_symmetric_part(cell::Cell{Data, Dim, T, L}, xc, degree
                             ) where {Data, Dim, T, L}
    @assert( length(cell.data.dx) > 0, "cell.data.dx is empty")
    @assert( length(cell.data.xref) > 0, "cell.data.xref is empty")
    num_nodes = size(xc,2)
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq_face = zeros(length(w1d)^(Dim-1))
    xq_face = zeros(Dim, length(wq_face))
    interp = zeros(length(wq_face), num_nodes)
    xref = cell.data.xref 
    dx = cell.data.dx
    E = zeros(num_nodes, num_nodes, Dim)
    for dir in ntuple(i -> i % 2 == 1 ? -div(i+1,2) : div(i,2), 2*Dim)
        rect = cell_side_rect(dir, cell)
        face_quadrature!(xq_face, wq_face, rect, x1d, w1d, abs(dir))
        build_interpolation!(interp, degree, xc, xq_face, xref, dx)
        for i in axes(interp,2)
            for j in axes(interp,2)
                for q in axes(interp,1)
                    E[i,j,abs(dir)] += interp[q,i] * wq_face[q] * interp[q,j] * sign(dir)
                end
            end
        end
    end
    return E
end

"""
    E = cell_symmetric_part(cell, xc, degree, levset [,fit_degree=degree])

Returns the symmetric part of the first-derivative SBP operator for the 
_possibly_ cut cell `cell`.  The point cloud associated with `cell` is `xc`, 
and the boundary operator uses `2*degree` exact quadrature.  The `fit_degree`
input indicates the degree of the Bernstein polynomial used by Algoim to
approximate the level set.
"""
function cell_symmetric_part(cell::Cell{Data, Dim, T, L}, xc, degree, levset;
                             fit_degree::Int=degree) where {Data, Dim, T, L}
    @assert( length(cell.data.dx) > 0, "cell.data.dx is empty")
    @assert( length(cell.data.xref) > 0, "cell.data.xref is empty")
    num_nodes = size(xc,2)
    xref = cell.data.xref 
    dx = cell.data.dx
    E = zeros(num_nodes, num_nodes, Dim)
    sumE = zeros(Dim)
    for dir in ntuple(i -> i % 2 == 1 ? -div(i+1,2) : div(i,2), 2*Dim)
        face = cell_side_rect(dir, cell)
        wq_face, xq_face = cut_face_quad(face, abs(dir), levset, degree+1,
                                         fit_degree=fit_degree)
        interp = zeros(length(wq_face), num_nodes)
        build_interpolation!(interp, degree, xc, xq_face, xref, dx)
        for i in axes(interp,2)
            for j in axes(interp,2)
                for q in axes(interp,1)
                    E[i,j,abs(dir)] += interp[q,i] * wq_face[q] * interp[q,j] * sign(dir)
                end
            end
        end
        sumE[abs(dir)] += sum(wq_face)*sign(dir)
    end

    # at this point, all planar faces of cell have been accounted for; now deal 
    # with the level-set surface `levset(x) = 0` passing through the cell
    surf_wts, surf_pts = cut_surf_quad(cell.boundary, levset, degree+1,
                                       fit_degree=fit_degree)

    if length(surf_wts) == 0
        # the cell was not actually cut, so the is nothing left to do but check
        for dir = 1:Dim
            @assert( abs(sumE[dir]) < 100*eps(), "geo. cons. law failed (1)" )
        end
        return E
    end

    # NOTE: negative sign needed because of sign convention in algoim
    surf_wts .*= -1.0
    interp = zeros(size(surf_wts,2), num_nodes)
    build_interpolation!(interp, degree, xc, surf_pts, xref, dx)
    for dir = 1:Dim
        for i in axes(interp,2)
            for j in axes(interp,2)
                for q in axes(interp,1)
                    E[i,j,dir] += interp[q,i] * surf_wts[dir,q] * interp[q,j]
                end
            end
        end
    end

    return E
end

"""
    make_compatible!(E, H, cell, xc, degree)

Modifies the SBP symmetric operator `E` for cell `cell` such that it is
compatible with the diagonal norm `H`.  The operators are defined over the
nodes `xc` and are for a degree `degree` exact SBP first-derivative operator.
"""
function make_compatible!(E, H, cell::Cell{Data, Dim, T, L}, xc, degree
                          ) where {Data, Dim, T, L}
    @assert( degree >= 0, "degree must be non-negative" )
    @assert( size(E,1) == size(E,2) == length(H) == size(xc,2), 
            "E, H, and/or xc are incompatible sizes" )
    @assert( size(xc,1) == Dim, "xc is incompatible with Dim" )

    num_basis = binomial(Dim + degree, Dim)
    num_nodes = size(xc,2)
    xref = cell.data.xref 
    dx = cell.data.dx

    # transform the points to local coordinates
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xref[I[1]])/dx[I[1]] - 0.5
    end

    # evaluate the polynomial basis and its derivatives at the nodes
    V = zeros(num_nodes, num_basis)
    dV = zeros(num_nodes, num_basis, Dim)
    workc = zeros((Dim+1)*num_nodes)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    poly_basis_derivatives!(dV, degree, xc_trans, Val(Dim))

    # construct the linear system that the symmetric matrix must satisfy
    num_sym_vars = div(num_nodes*(num_nodes+1),2)
    num_eqns = div(num_basis*(num_basis+1),2)
    A = zeros(num_eqns, num_sym_vars)
    B = zeros(num_eqns, Dim)
    eqn_idx = 1
    var_idx = 1
    # i and j define the loop over equations
    for i = 1:num_basis 
        for j = 1:i
            # row and col define the loop over entries in E
            for row = 1:num_nodes
                offset = div((row-1)*row,2)
                for col = 1:row-1
                    A[eqn_idx, offset+col] += V[row,i]*V[col,j]
                    A[eqn_idx, offset+col] += V[col,i]*V[row,j]
                end 
                A[eqn_idx, offset+row] += V[row,i]*V[row,j]
            end
            for d = 1:Dim 
                B[eqn_idx,d] = dot(dV[:,i,d], H.*V[:,j]) + 
                                dot(dV[:,j,d], H.*V[:,i])
                B[eqn_idx,d] /= dx[d]
                B[eqn_idx,d] -= dot(V[:,i], E[:,:,d] * V[:,j])
            end
            eqn_idx += 1
        end
    end

    # solve for the min norm solution
    vals = zeros(num_sym_vars)
    for d = 1:Dim
        solve_min_norm!(vals, A', vec(B[:,d]))
        for row = 1:num_nodes 
            offset = div((row-1)*row,2)
            for col = 1:row-1
                E[row,col,d] += vals[offset+col]
                E[col,row,d] += vals[offset+col]
            end
            E[row,row,d] += vals[offset+row]
        end
    end
    return nothing
end

"""
    S = cell_skew_part(E, H, cell, xc, degree)

Returns the skew-symmetric parts of SBP diagonal-norm operators for the element
`cell` based on the nodes `xc`.  The operator is exact for polynomials of total 
degree `degree`.  The diagonal norm for the cell is provided in the array `H`, 
which must be exact for degree `2*degree - 1` polynomials over `xc`.  Finally,
the symmetric part of the SBP operators must be provided in `E`.  Note that `E`
and the returned `S` are three dimensional arrays, with `E[:,:,d]` and
`S[:,:,d]` holding the operators for the direction `d`.

**NOTE**: E and H must be compatible, in the sense of SBP operators.
"""
function cell_skew_part(E, H, cell::Cell{Data, Dim, T, L}, xc, degree
                        ) where {Data, Dim, T, L}

    num_basis = binomial(Dim + degree, Dim)
    num_nodes = size(xc,2)

    xref = cell.data.xref 
    dx = cell.data.dx
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xref[I[1]])/dx[I[1]] - 0.5
    end

    V = zeros(num_nodes, num_basis)
    dV = zeros(num_nodes, num_basis, Dim)
    workc = zeros((Dim+1)*num_nodes)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    poly_basis_derivatives!(dV, degree, xc_trans, Val(Dim))

    # construct the linear system that the skew matrices must satisfy 
    num_skew_vars = div(num_nodes*(num_nodes-1),2)
    num_eqns = num_nodes*num_basis
    A = zeros(num_eqns, num_skew_vars)
    B = zeros(num_eqns, Dim)
    ptr = 0
    for k = 1:num_basis
        for row = 2:num_nodes 
            offset = div((row-1)*(row-2),2)
            for col = 1:row-1
                A[ptr+row,offset+col] += V[col,k]
                A[ptr+col,offset+col] -= V[row,k]
            end 
        end
        for d = 1:Dim
            # the factor of 1/dx[d] accounts for the transformation above
            B[ptr+1:ptr+num_nodes,d] = diagm(H)*dV[:,k,d]/dx[d] - 0.5*E[:,:,d]*V[:,k]
        end
        ptr += num_nodes
    end
    S = zeros(num_nodes, num_nodes, Dim)
    vals = zeros(num_skew_vars)
    for d = 1:Dim
        solve_min_norm!(vals, A', vec(B[:,d]))
        for row = 2:num_nodes 
            offset = div((row-1)*(row-2),2)
            for col = 1:row-1
                S[row,col,d] += vals[offset+col]
                S[col,row,d] -= vals[offset+col]
            end
        end
        #println("norm(A*vals - vec(B[:,d])) = ", norm(A*vals - vec(B[:,d])))
    end
    return S
end

"""
    Sface = interface_skew_part(face, xc_left, xc_right, degree)

Constructs the form

``\\int_{\\text{face}} V_i V_j d\\Gamma``

where the integral is over the face `face` with unit normal in the coordinate
direction `face.dir`.  The functions ``V_i`` and ``V_j`` can be regarded as
degree `degree` basis functions at the nodes `i` and `j` within the stencil of
the left and right cells, respectively.
"""
function interface_skew_part(face::Face{Dim, T, Cell}, xc_left, xc_right, degree
                             ) where {Dim,T,Cell}
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq_face = zeros(length(w1d)^(Dim-1))
    xq_face = zeros(Dim, length(wq_face))
    face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)
    interp_left = zeros(length(wq_face), size(xc_left,2))
    interp_right = zeros(length(wq_face), size(xc_right,2))
    build_interpolation!(interp_left, degree, xc_left, xq_face,
                         face.cell[1].data.xref, face.cell[1].data.dx)
    build_interpolation!(interp_right, degree, xc_right, xq_face,
                         face.cell[2].data.xref, face.cell[2].data.dx)
    Sface = zeros(size(xc_left,2), size(xc_right,2))
    for i in axes(interp_left,2)
        for j in axes(interp_right,2)
            for (q, wq) in enumerate(wq_face)
                Sface[i,j] += interp_left[q,i] * interp_right[q,j] * wq
            end
            Sface[i,j] *= 0.5
        end
    end
    return Sface
end

"""
    Sface = interface_skew_part(face, xc_left, xc_right, degree, levset
                                [, fit_degree=degree])

This version of the method is for planar interfaces that are cut by a level-set
geometry defined by the function `levset`.  The optional kwarg `fit_degree`
indicates the degree of the Bernstein polynomials used to approximate the
level-set within the Algoim library.
"""
function interface_skew_part(face::Face{Dim, T, Cell}, xc_left, xc_right,
                             degree, levset; fit_degree::Int=fit_degree
                             ) where {Dim,T,Cell}
    wq_face, xq_face = cut_face_quad(face.boundary, face.dir, levset, degree+1,
                                     fit_degree=fit_degree)
    interp_left = zeros(length(wq_face), size(xc_left,2))
    interp_right = zeros(length(wq_face), size(xc_right,2))
    build_interpolation!(interp_left, degree, xc_left, xq_face,
                         face.cell[1].data.xref, face.cell[1].data.dx)
    build_interpolation!(interp_right, degree, xc_right, xq_face,
                         face.cell[2].data.xref, face.cell[2].data.dx)
    Sface = zeros(size(xc_left,2), size(xc_right,2))
    for i in axes(interp_left,2)
        for j in axes(interp_right,2)
            for (q, wq) in enumerate(wq_face)
                Sface[i,j] += interp_left[q,i] * interp_right[q,j] * wq
            end
            Sface[i,j] *= 0.5
        end
    end
    return Sface
end

"""
    S = skew_operator(root, ifaces, xc, levset, degree [, fit_degree=degree])

Constructs the skew-symmetric part of a (global), first-derivative SBP 
operator.  The integration mesh is given by `root` and `xc` defines the cloud 
of distributed nodes where the degrees of freedom are stored.  `ifaces` is an 
array of interfaces (not boundary faces) corresponding to `root`.  `levset` is 
a function that defines the immersed geomtry, if any.  The skew-symmetric 
matrix is degree `degree` exact.  Finally, `fit_degree` gives the polynomial 
degree of the Bernstein polynomials used to approximate `levset` by the Algoim 
library.
"""
function skew_operator(root::Cell{Data, Dim, T, L}, ifaces, xc, levset, degree;
                       fit_degree::Int=degree) where {Data, Dim, T, L}
    # set up arrays to store sparse matrix information
    rows = Array{Array{Int64}}(undef, Dim)
    cols = Array{Array{Int64}}(undef, Dim)
    Svals = Array{Array{T}}(undef, Dim)
    for d = 1:Dim
        rows[d] = Int[]
        cols[d] = Int[] 
        Svals[d] = T[]
    end

    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        # get the nodes in this cell's stencil
        nodes = view(xc, :, cell.data.points)
        xref = cell.data.xref 
        dx = cell.data.dx
        moments = cell.data.moments
        Hcell = cell_quadrature(2*degree-1, nodes, moments, xref, dx, Val(Dim))

        if is_cut(cell)
            # this cell *may* be cut; use Saye's algorithm
            Ecell = cell_symmetric_part(cell, nodes, degree, levset, fit_degree=fit_degree)
            make_compatible!(Ecell, Hcell, cell, xc, degree)
        else
            # this cell is not cut
            Ecell = cell_symmetric_part(cell, nodes, degree) 
        end
        Scell = cell_skew_part(Ecell, Hcell, cell, nodes, degree)

        # Now load into sparse-matrix arrays
        for (i,row) in enumerate(cell.data.points)            
            for (j,col) in enumerate(cell.data.points[i+1:end])                
                for d = 1:Dim
                    if abs(Scell[i,i+j,d]) > 1e-13
                        append!(rows[d], row)
                        append!(cols[d], col)
                        append!(Svals[d], Scell[i,i+j,d])
                    end
                end
            end
        end
    end

    # loop over interfaces and add contributions to skew-symmetric matrix
    for face in ifaces
        if is_immersed(face)
            continue
        end
        # get the nodes for the two adjacent cells
        xc_left = view(xc, :, face.cell[1].data.points)
        xc_right = view(xc, :, face.cell[2].data.points)

        if is_cut(face)
            # this face *may* be cut; use Saye's algorithm
            Sface = interface_skew_part(face, xc_left, xc_right, degree, levset,
                                        fit_degree=fit_degree)
        else
            # this face is not cut
            Sface = interface_skew_part(face, xc_left, xc_right, degree)
        end
        
        # load into sparse-matrix arrays
        for (i,row) in enumerate(face.cell[1].data.points)
            for (j,col) in enumerate(face.cell[2].data.points)
                if col == row continue end 
                if abs(Sface[i,j]) > 1e-13
                    append!(rows[face.dir], row)    
                    append!(cols[face.dir], col)
                    append!(Svals[face.dir], Sface[i,j])
                end
            end
        end
    end

    S = SVector(ntuple(d -> sparse(rows[d], cols[d], Svals[d]), Dim))
    return S
end

"""
    add_face_to_boundary!(bndry, face, xc, degree)

Computes the quadrature points, normal vector, degrees of freedom, and 
interpolation operator for the face `face` and adds this data to the given 
`BoundaryOperator`, `bndry`.  `xc` are the locations of the nodes in the 
stencil of `face`, and `degree` determines the order of accuracy of the 
quadrature (`2*degree+1`) and the interpolation.
"""
function add_face_to_boundary!(bndry::BoundaryOperator{T}, face, xc, degree
                               ) where {T}
    cell = face.cell[1]
    Dim = size(xc,1)
    @assert( length(cell.data.points) == size(xc,2), 
            "face.cell[1] and xc are incompatible")
    x1d, w1d = lg_nodes(degree+1)
    num_nodes = length(cell.data.points)
    num_quad = length(w1d)^(Dim-1)
    wq_face = zeros(num_quad)
    xq_face, nrm, dof, prj = push_new_face!(bndry, Dim, num_nodes, num_quad)
    face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)
    build_interpolation!(prj, degree, xc, xq_face, cell.data.xref, cell.data.dx)
    for (q,w) in enumerate(wq_face) 
        nrm[abs(face.dir),q] = sign(face.dir)*w 
    end
    dof[:] = cell.data.points # just make a reference to points?
    return nothing
end

"""
    add_face_to_boundary!(bndry, face, xc, degree, levset [, fit_degree=degree])

This version of the method is for planar boundary faces that are cut by the
level-set geometry defined by the function `levset`.  The optional kwarg 
`fit_degree` indicates the degree of the Bernstein polynomials used to 
approximate the level-set within the Algoim library.
"""
function add_face_to_boundary!(bndry::BoundaryOperator{T}, face, xc, degree, 
                               levset; fit_degree::Int=degree
                               ) where {T}
    cell = face.cell[1]
    @assert( length(cell.data.points) == size(xc,2), 
            "face.cell[1] and xc are incompatible")
    wq_cut, xq_cut = cut_face_quad(face.boundary, face.dir, levset, degree+1,
                                     fit_degree=fit_degree)
    num_nodes = length(cell.data.points)
    num_quad = size(xq_cut,2)
    xq_face, nrm, dof, prj = push_new_face!(bndry, Dim, num_nodes, num_quad)
    xq_face[:,:] = xq_cut[:,:]
    build_interpolation!(prj, degree, xc, xq_face, cell.data.xref, cell.data.dx)
    for (q,w) in enumerate(wq_cut) 
        nrm[abs(face.dir),q] = sign(face.dir)*w 
    end
    dof[:] = cell.data.points # just make a reference to points?
    return nothing
end

function add_face_to_boundary!(bndry::BoundaryOperator{T},
                               cell::Cell{Data, Dim, T, L}, xc, degree, levset; fit_degree::Int=degree) where {Data, Dim, T, L}
    @assert( length(cell.data.points) == size(xc,2), 
            "cell and xc are incompatible")
    surf_wts, surf_pts = cut_surf_quad(cell.boundary, levset, degree+1,
                                       fit_degree=fit_degree)
    num_quad = size(surf_pts,2)
    if num_quad == 0
        # Algoim may determine the cell is not actually cut
        return nothing 
    end
    num_nodes = length(cell.data.points)
    xq_face, nrm, dof, prj = push_new_face!(bndry, Dim, num_nodes, num_quad)
    xq_face[:,:] = surf_pts[:,:]
    build_interpolation!(prj, degree, xc, xq_face, cell.data.xref, cell.data.dx)
    nrm[:,:] = -surf_wts
    dof[:] = cell.data.points # just make a reference to points?
    return nothing
end


function boundary_operators(bc_map, root::Cell{Data, Dim, T, L}, boundary_faces,
                            xc, levset, degree; fit_degree::Int=degree)
                            where {Data, Dim, T, L}

    # Create a boundary operator for each unique BC
    bc_types = unique(values(bc_map))
    E = Dict( bc => BoundaryOperator(eltype(xc)) for bc in bc_types)

    # loop over all planar boundary faces.
    for face in boundary_faces
        if is_immersed(face)
            continue
        end
        di = abs(face.dir)
        side = 2*di + div(sign(face.dir) - 1,2)
        if is_cut(face)
            add_face_to_boundary!(E[bc_map[side]], face, 
                                  view(xc, :, face.cell[1].data.points), degree,
                                  levset, fit_degree=fit_degree)
        else
            add_face_to_boundary!(E[bc_map[side]], face, 
                                  view(xc, :, face.cell[1].data.points), degree)
        end
    end

    # loop over the non-planar boundary faces
    for cell in allleaves(root)
        if is_cut(cell)
            add_face_to_boundary!(E[bc_map["ib"]], cell, 
                                  view(xc, :, cell.data.points), degree,
                                  levset, fit_degree=fit_degree)
        end
    end

    return E
end


function build_boundary_operator(root::Cell{Data, Dim, T, L}, boundary_faces, 
                                 xc, degree) where {Data, Dim, T, L}
    num_face = length(boundary_faces)
    bnd_nrm = Array{Matrix{T}}(undef, num_face)
    bnd_pts = Array{Matrix{T}}(undef, num_face)
    bnd_dof = Array{Vector{Int}}(undef, num_face)
    bnd_prj = Array{Matrix{T}}(undef, num_face)

    # find the maximum number of phi basis over all cells
    max_basis = 0
    for cell in allleaves(root)
        max_basis = max(max_basis, length(cell.data.points))
    end

    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(w1d)^(Dim-1)
    wq_face = zeros(num_quad)
    
    #work = dgd_basis_work_array(degree, max_basis, length(wq_face), Val(Dim))
    #work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq_face))

    # loop over boundary faces 
    for (findex, face) in enumerate(boundary_faces)
        di = abs(face.dir)
        # get the Gauss points on face
        bnd_pts[findex] = zeros(Dim, num_quad)
        face_quadrature!(bnd_pts[findex], wq_face, face.boundary, x1d, w1d, di)
        # evaluate the basis functions on the face nodes; this defines prolong
        num_nodes = length(face.cell[1].data.points)
        bnd_prj[findex] = zeros(num_quad, num_nodes)
        build_interpolation!(bnd_prj[findex], degree,
                             view(xc, :, face.cell[1].data.points),
                             bnd_pts[findex], face.cell[1].data.xref,
                             face.cell[1].data.dx)

        #dgd_basis!(bnd_prj[findex], degree,
        #           view(points, :, face.cell[1].data.points),
        #           bnd_pts[findex], work, Val(Dim))
        # define the face normals
        bnd_nrm[findex] = zero(bnd_pts[findex])
        for q = 1:num_quad
            bnd_nrm[findex][di,q] = sign(face.dir)*wq_face[q]
        end
        # get the degrees of freedom 
        bnd_dof[findex] = deepcopy(face.cell[1].data.points)
    end
    return bnd_pts, bnd_nrm, bnd_dof, bnd_prj
end

# function build_boundary_operator(root::Cell{Data, Dim, T, L}, boundary_faces, 
#                                  points, degree) where {Data, Dim, T, L}
#     num_face = length(boundary_faces)
#     bnd_nrm = Array{Matrix{T}}(undef, num_face)
#     bnd_pts = Array{Matrix{T}}(undef, num_face)
#     bnd_dof = Array{Vector{Int}}(undef, num_face)
#     bnd_prj = Array{Matrix{T}}(undef, num_face)

#     # find the maximum number of phi basis over all cells
#     max_basis = 0
#     for cell in allleaves(root)
#         max_basis = max(max_basis, length(cell.data.points))
#     end

#     x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
#     num_quad = length(w1d)^(Dim-1)
#     wq_face = zeros(num_quad)
#     #work = dgd_basis_work_array(degree, max_basis, length(wq_face), Val(Dim))
#     work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq_face))

#     # loop over boundary faces 
#     for (findex, face) in enumerate(boundary_faces)
#         di = abs(face.dir)
#         # get the Gauss points on face
#         bnd_pts[findex] = zeros(Dim, num_quad)
#         face_quadrature!(bnd_pts[findex], wq_face, face.boundary, x1d, w1d, di)
#         # evaluate the basis functions on the face nodes; this defines prolong
#         num_basis = length(face.cell[1].data.points)
#         bnd_prj[findex] = zeros(num_quad, num_basis)
#         dgd_basis!(bnd_prj[findex], degree,
#                    view(points, :, face.cell[1].data.points),
#                    bnd_pts[findex], work, Val(Dim))
#         # define the face normals
#         bnd_nrm[findex] = zero(bnd_pts[findex])
#         for q = 1:num_quad
#             bnd_nrm[findex][di,q] = sign(face.dir)*wq_face[q]
#         end
#         # get the degrees of freedom 
#         bnd_dof[findex] = deepcopy(face.cell[1].data.points)
#     end
#     return bnd_pts, bnd_nrm, bnd_dof, bnd_prj
# end

function weak_differentiate!(dudx, u, di, sbp)
    fill!(dudx, 0)
    # first apply the skew-symmetric part of the operator 
    dudx[:] += sbp.S[di]*u 
    dudx[:] -= sbp.S[di]'*u
    # next apply the symmetric part of the operator 
    for (f, Pface) in enumerate(sbp.bnd_prj)
        nrm = sbp.bnd_nrm[f]
        for i = 1:size(Pface,2)
            row = sbp.bnd_dof[f][i]
            for j = 1:size(Pface,2)
                col = sbp.bnd_dof[f][j]
                for q = 1:size(Pface,1)                    
                    dudx[row] += 0.5*Pface[q,i]*nrm[di,q]*Pface[q,j]*u[col]
                end
            end
        end
    end
end