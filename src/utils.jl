# Various functions that do not fit nicely elsewhere 

"""
    solve_min_norm!(w, V, b)

Finds the minimum norm solution to `transpose(V) w = b`.
"""
function solve_min_norm!(w, V, b)
    F = qr(V)
    Q = Matrix(F.Q) # If using Q directly, restrict to first length(b) cols 
    w .= Q * (F.R' \ b)
    return nothing
end

"""
    solve_min_norm_diff!(w, dw, V, dV, b)

Forward mode of `solve_min_norm!` treating `b` as constant.  This is useful
for verifying the reverse mode implementation.
"""
function solve_min_norm_diff!(w, dw, V, dV, b)
    F = qr(V)
    Q = Matrix(F.Q)
    Rb = F.R' \ b
    w .= Q * Rb
    QdV = Q'*dV
    QdVR = QdV / F.R 
    X = tril(QdVR,-1)
    X = X - X'
    dQ = Q*X + dV / F.R - Q*QdVR
    dR = QdV - X*F.R
    dw .= dQ * Rb - Q * (F.R' \ (dR' * Rb) )
    return nothing 
end

"""
    solve_min_norm_rev!(V_bar, w_bar, V, b)

Reverse mode of `solve_min_norm!`.  On entry, `w_bar` holds the derivatives of
the objective w.r.t. the weights.  On exit, `V_bar` holds the derivatives of the
objective w.r.t. the matrix `V`.
"""
function solve_min_norm_rev!(V_bar, w_bar, V, b)
    F = qr(V)
    Q = Matrix(F.Q)
    Rtb = F.R' \ b
    # w .= Q * Rtb 
    Q_bar = w_bar * Rtb'
    Rtb_bar = Q' * w_bar
    # Rtb = Rt \ b    
    R_bar = -(F.R \ Rtb_bar * Rtb')'
    M = R_bar*F.R' - Q'*Q_bar
    M = triu(M) + transpose(triu(M,1))
    V_bar .= (Q_bar + Q * M) / F.R'
    return nothing
end

"""
    build_interpolation!(interp, degree, x, x_interp, xref, dx)

Creates the matrix `interp` that performs interpolation from nodes `x` to
points `x_interp`, which is `degree` total polynomial degree exact.  The 
arrays `xref` and `dx` are used to shift and scale the coordinates to 
improve the conditioning of the Vandermonde matrix.

**Note:** This routine assumes that `x` is unisolvent.  Also, the matrix 
`interp` is the minmum-norm solution when there are more nodes in `x` than 
total `degree` basis functions.
"""
function build_interpolation!(interp, degree, x, x_interp, xref, dx)
    @assert( size(x,1) == size(x_interp,1), "x and x_interp are inconsistent")    
    Dim = size(x,1)
    num_nodes = size(x,2)
    num_basis = binomial(Dim + degree, Dim)
    @assert( num_nodes >= num_basis, "too few nodes in x for degree exactness")
    num_interp = size(x_interp,2)
    x_trans = zero(x)
    for I in CartesianIndices(x)
        x_trans[I] = (x[I] - xref[I[1]])/dx[I[1]] - 0.5
    end
    x_interp_trans = zero(x_interp)
    for I in CartesianIndices(x_interp)
        x_interp_trans[I] = (x_interp[I] - xref[I[1]])/dx[I[1]] - 0.5
    end
    work = zeros(eltype(x), (Dim+1)*num_nodes)
    V = zeros(eltype(x), num_nodes, num_basis)
    poly_basis!(V, degree, x_trans, work, Val(Dim))
    work_interp = zeros(eltype(x_interp), (Dim+1)*num_interp)
    V_interp = zeros(eltype(x), num_interp, num_basis)
    poly_basis!(V_interp, degree, x_interp_trans, work_interp, Val(Dim))
    solve_min_norm!(interp', V, V_interp')
    return nothing
end
