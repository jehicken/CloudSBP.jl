
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
            # @assert( abs(sumE[dir]) < 10^degree*eps(),
            #         "geo. cons. law failed (1)" )
            if abs(sumE[dir]) > 10^degree*eps()
                println("abs(sumE[",dir,"] = ",abs(sumE[dir]))
            end
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
    E = boundary_sym_part(face, xc, degree)

Returns the symmetric boundary form 

``\\int_{\\text{face}} V_i V_j n_{\text{dir}} d\\Gamma``

where the integral is over the face `face` with unit normal in the coordinate
direction `face.dir`.  The functions ``V_i`` and ``V_j`` can be regarded as
degree `degree` basis functions at the nodes `i` and `j` within the stencil of
the cell whose stencil's coordinates are `xc`.
"""
function boundary_sym_part(face, xc, degree)
    cell = face.cell[1]
    Dim = size(xc,1)
    num_nodes = length(cell.data.points)
    @assert( num_nodes == size(xc,2), "face.cell[1] and xc are incompatible")
    x1d, w1d = lg_nodes(degree+1)
    num_quad = length(w1d)^(Dim-1)
    wq_face = zeros(num_quad)
    xq_face = zeros(Dim, num_quad)
    interp = zeros(num_quad, num_nodes)
    face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, abs(face.dir))
    build_interpolation!(interp, degree, xc, xq_face, cell.data.xref,
                         cell.data.dx)
    E = zeros(num_nodes, num_nodes)
    for i in axes(interp,2)
        for j in axes(interp,2)
            for q in axes(interp,1)
                E[i,j] += interp[q,i]*wq_face[q]*interp[q,j]*sign(face.dir)
            end
        end
    end
    return E
end

"""
    E = boundary_sym_part(face, xc, degree, levset [, fit_degree=degree])

This version of the method is for planar boundary faces that are cut by the
level-set geometry defined by the function `levset`.  The optional kwarg 
`fit_degree` indicates the degree of the Bernstein polynomials used to 
approximate the level-set within the Algoim library.
"""
function boundary_sym_part(face, xc, degree, levset; fit_degree::Int=degree
                           )
    cell = face.cell[1]
    num_nodes = length(cell.data.points)
    @assert( num_nodes == size(xc,2), "face.cell[1] and xc are incompatible")
    wq_cut, xq_cut = cut_face_quad(face.boundary, abs(face.dir), levset,
                                   degree+1, fit_degree=fit_degree)
    interp = zeros(length(wq_cut), num_nodes)
    build_interpolation!(interp, degree, xc, xq_cut, cell.data.xref,
                         cell.data.dx)
    E = zeros(num_nodes, num_nodes)
    for i in axes(interp,2)
        for j in axes(interp,2)
            for q in axes(interp,1)
                E[i,j] += interp[q,i]*wq_cut[q]*interp[q,j]*sign(face.dir)
            end
        end
    end
    return E
end

"""
    E = boundary_sym_part!(cell, xc, degree, levset [, fit_degree=degree])

This version of the method is for faces that are the intersection between the
level-set `levset(x)=0` and the cell `cell`.  As usual, `fit_degree` indicates
the degree of the Bernstein polynomials used to approximate the level-set
within the Algoim library.
"""
function boundary_sym_part(cell::Cell{Data, Dim, T, L}, xc, degree, levset;
                           fit_degree::Int=degree) where {Data, Dim, T, L}
    num_nodes = length(cell.data.points)
    @assert( num_nodes == size(xc,2), "cell and xc are incompatible")
    surf_wts, surf_pts = cut_surf_quad(cell.boundary, levset, degree+1,
                                       fit_degree=fit_degree)
    num_quad = size(surf_pts,2)
    E = zeros(num_nodes, num_nodes, Dim)
    if num_quad == 0
        # Algoim may determine the cell is not actually cut
        return E
    end
    # NOTE: negative sign needed because of sign convention in algoim
    surf_wts .*= -1.0
    interp = zeros(num_quad, num_nodes)
    build_interpolation!(interp, degree, xc, surf_pts, cell.data.xref,
                         cell.data.dx)
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
    E = symmetric_operator(root, bfaces, xc, levset, degree [, fit_degree=degree])

Constructs the symmetric part of a (global), first-derivative SBP operators as
an `SVector` of sparse matrices; only the symmetric part is stored. The
integration mesh is given by `root` and `xc` defines the cloud of distributed
nodes where the degrees of freedom are stored.  `bfaces` is an array of boundary
faces corresponding to `root`.  `levset` is a function that defines the immersed
geomtry, if any.  The symmetric matrix is degree `degree` exact.  Finally,
`fit_degree` gives the polynomial degree of the Bernstein polynomials used to
approximate `levset` by the Algoim library.

**Note**: The sparse matrices are not useful for much other than verifying the
accuracy of the global skew-symmetric parts; this is because we do not
distinguish different boundaries, nor do we provide the decomposition of the
symmetric part into interpolation operators and face quadrature.
"""
function symmetric_operator(root::Cell{Data, Dim, T, L}, bfaces, xc, levset,
                            degree; fit_degree::Int=degree
                            ) where {Data, Dim, T, L}
    # set up arrays to store sparse matrix information
    rows = Array{Array{Int64}}(undef, Dim)
    cols = Array{Array{Int64}}(undef, Dim)
    Evals = Array{Array{T}}(undef, Dim)
    for d = 1:Dim
        rows[d] = Int[]
        cols[d] = Int[] 
        Evals[d] = T[]
    end

    # loop over all planar boundary faces
    for face in bfaces
        if is_immersed(face)
            continue
        end

        if is_cut(face)
            Eface = boundary_sym_part(face, view(xc, :, face.cell[1].data.points),
                                      degree, levset, fit_degree=fit_degree)
        else
            Eface = boundary_sym_part(face, view(xc, :, face.cell[1].data.points),
                                      degree)
        end
        d = abs(face.dir)
        # Load into sparse-matrix arrays
        for (i,row) in enumerate(face.cell[1].data.points)
            for (j,col) in enumerate(face.cell[1].data.points[1:i])
                if abs(Eface[i,j]) > 1e-13
                    append!(rows[d], row)
                    append!(cols[d], col)
                    append!(Evals[d], Eface[i,j])
                end
            end
        end
    end

    # loop over the non-planar boundary faces
    for cell in allleaves(root)
        if !is_cut(cell)
            continue
        end
        nodes = view(xc, :, cell.data.points)
        xref = cell.data.xref 
        dx = cell.data.dx
        moments = cell.data.moments
        Hcell = cell_quadrature(2*degree-1, nodes, moments, xref, dx, Val(Dim))

        # Need to "jump some hoops" to get the compatible boundary operator
        Ecell = cell_symmetric_part(cell, nodes, degree, levset,
                                    fit_degree=fit_degree)
        E0 = deepcopy(Ecell)
        make_compatible!(Ecell, Hcell, cell, nodes, degree)

        # check compatibility 
        # !!!!!!!!!!!!!!!!!!!!!!!!!
        num_nodes = size(nodes,2)
        num_basis = binomial(Dim + degree, Dim)
        V = zeros(num_nodes, num_basis)
        monomial_basis!(V, degree, nodes, Val(Dim))
        dV = zeros(num_nodes, num_basis, Dim)
        monomial_basis_derivatives!(dV, degree, nodes, Val(Dim))
        for d = 1:Dim
            VtEV = V'*Ecell[:,:,d]*V 
            VtHdV = V'*(Hcell.*dV[:,:,d]) + dV[:,:,d]'*(Hcell.*V)
            println("norm(VtHdV - VtEV) = ",norm(VtHdV - VtEV))
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!


        Esurf = boundary_sym_part(cell, nodes, degree, levset,
                                  fit_degree=fit_degree)
        Esurf[:,:,:] += Ecell - E0 

        println("cell.boundary.origin = ",cell.boundary.origin)
        println("\tsum(H) = ",sum(Hcell))
        println("\tEsurf = ",Esurf[:,:,2])

        # Load into sparse-matrix arrays
        for (i,row) in enumerate(cell.data.points)
            for (j,col) in enumerate(cell.data.points[1:i])
                for d = 1:Dim
                    if abs(Esurf[i,j,d]) > 1e-13
                        append!(rows[d], row)
                        append!(cols[d], col)
                        append!(Evals[d], Esurf[i,j,d])
                    end
                end
            end
        end
    end

    E = SVector(ntuple(d -> sparse(rows[d], cols[d], Evals[d]), Dim))
    return E
end