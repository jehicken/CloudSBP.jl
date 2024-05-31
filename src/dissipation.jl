"""
Face-based dissipation operator

`w_face` are the interface areas, `x_face` are the face centers, `R_left` and 
`R_right` are interpolation operators to the faces.
"""
mutable struct Dissipation{T}
    dir::Vector{Int}
    w_face::Vector{T}
    x_face::Matrix{T}
    R_left::SparseMatrixCSC{T, Int64}
    R_right::SparseMatrixCSC{T, Int64}
end

"""
    Rl, Rr, x, w  = interface_interp(face, xc_left, xc_right, degree)

Returns interpolation operators `Rl` and `Rr` from `xc_left` and `xc_right` to `x`, respectively, where `x` is the returned averaged center of the face `face`.  `w` is the weight associate with `face` (its area).
"""
function interface_interp(face::Face{Dim, T, Cell}, xc_left, xc_right, degree
                          ) where {Dim,T,Cell}
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq_face = zeros(length(w1d)^(Dim-1))
    xq_face = zeros(Dim, length(wq_face))
    face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)
    # average to get the face center and weight 
    w = sum(wq_face)
    x = zeros(Dim)
    for (q, wq) in enumerate(wq_face)
        x[:] += xq_face[:,q]*wq
    end
    x ./= w 
    # now get the interpolations 
    Rl = zeros(size(xc_left,2))
    Rr = zeros(size(xc_right,2))
    build_interpolation!(reshape(Rl,1,:), degree, xc_left, reshape(x, (Dim,1)),
                         face.cell[1].data.xref, face.cell[1].data.dx)
    build_interpolation!(reshape(Rr,1,:), degree, xc_right, reshape(x, (Dim,1)),
                         face.cell[2].data.xref, face.cell[2].data.dx)
    return Rl, Rr, x, w
end

"""
    Rl, Rr, x, w  = interface_interp(face, xc_left, xc_right, degree, levset
                                     [, fit_degree=degree])

This version of the method is for planar interfaces that are cut by a level-set
geometry defined by the function `levset`.  The optional kwarg `fit_degree`
indicates the degree of the Bernstein polynomials used to approximate the
level-set within the Algoim library.
"""
function interface_interp(face::Face{Dim, T, Cell}, xc_left, xc_right, degree,
                          levset; fit_degree::Int=degree) where {Dim,T,Cell}
    wq_face, xq_face = cut_face_quad(face.boundary, face.dir, levset, degree+1,
                                     fit_degree=fit_degree)
    Rl = zeros(size(xc_left,2))
    Rr = zeros(size(xc_right,2))
    x = zeros(Dim)
    if isempty(wq_face)
        w = 0.0
        return Rl, Rr, x, w
    end
    # average to get the face center and weight 
    w = sum(wq_face)
    for (q, wq) in enumerate(wq_face)
        x[:] += xq_face[:,q]*wq
    end
    x ./= w 
    # now get the interpolations 
    build_interpolation!(reshape(Rl,1,:), degree, xc_left, reshape(x, (Dim,1)),
                         face.cell[1].data.xref, face.cell[1].data.dx)
    build_interpolation!(reshape(Rr,1,:), degree, xc_right, reshape(x, (Dim,1)),
                         face.cell[2].data.xref, face.cell[2].data.dx)
    return Rl, Rr, x, w
end

"""
    face_diss = build_dissipation(ifaces, xc, degree, levset,
                                  [, fit_degree=degree])

Returns a `Dissipation` type that can be used for artificial dissipation over a 
point cloud.  `iface` is a list of interfaces, and `xc[:,i]` are the 
coordinates of the ith point in the cloud.  `degree` is the polynomial degree 
for which the interpolations to the faces is exact.  `levset` defines a 
level-set geometry, and the optional kwarg `fit_degree` indicates the degree of 
the Bernstein polynomials used to approximate the level-set within the Algoim 
library.
"""
function build_dissipation(ifaces::Vector{Face{Dim, T, Cell}}, xc, degree,
                           levset; fit_degree::Int=degree) where {Dim, T, Cell}

    # count the total number of faces that are not immersed
    num_face = length(ifaces) - number_immersed(ifaces)
    dir = zeros(Int, num_face)
    w_face = zeros(T, num_face)
    x_face = zeros(T, Dim, num_face)
    rows_left = Int[]
    cols_left = Int[]
    Dvals_left = T[]
    rows_right = Int[]
    cols_right = Int[]
    Dvals_right = T[]

    # loop over interfaces and construct interpolation operators and weights
    f = 0
    for face in ifaces
        if is_immersed(face)
            continue
        end
        f += 1
        dir[f] = face.dir
        # get the nodes for the two adjacent cells
        xc_left = view(xc, :, face.cell[1].data.points)
        xc_right = view(xc, :, face.cell[2].data.points)

        if is_cut(face)
            # this face *may* be cut; use Saye's algorithm
            R_left, R_right, x_face[:,f], w_face[f] = 
                interface_interp(face, xc_left, xc_right, degree, levset,
                                 fit_degree=fit_degree)
        else
            # this face is not cut
            R_left, R_right, x_face[:,f], w_face[f] = 
                interface_interp(face, xc_left, xc_right, degree)
        end
        # load into sparse-matrix arrays
        for (i,col_left) in enumerate(face.cell[1].data.points)
            append!(rows_left, f)
            append!(cols_left, col_left)
            append!(Dvals_left, R_left[i])
        end
        for (j,col_right) in enumerate(face.cell[2].data.points)
            append!(rows_right, f)
            append!(cols_right, col_right)
            append!(Dvals_right, R_right[j])
        end
    end

    return Dissipation(dir, w_face, x_face,
                       sparse(rows_left, cols_left, Dvals_left),
                       sparse(rows_right, cols_right, Dvals_right))
end