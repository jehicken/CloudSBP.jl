
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