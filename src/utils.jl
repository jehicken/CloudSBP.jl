# Various functions that do not fit nicely elsewhere 

"""
    get_complex_step(T)

returns an appropriate complex-step size for the given type
"""
get_complex_step(::Type{T}) where {T <: Float32} = 1f-20
get_complex_step(::Type{T}) where {T <: Float64} = 1e-60
get_complex_step(::Type{T}) where {T <: ComplexF32} = 1f-20
get_complex_step(::Type{T}) where {T <: ComplexF64} = 1e-60

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


# function surf_quad_res!(res, pts, wts, degree, V, work, moments)
#     @assert( degree >= 0, "degree must be non-negative" )
#     @assert( size(pts,2) == length(wts), "pts inconsistent with wts" )
#     @assert( length(res) == length(moments), "res inconsistent with moments" )
#     Dim = size(pts,1)
#     num_quad = size(pts,2)
#     @assert( size(V,1) == num_quad, "V inconsisent with pts" )
#     @assert( size(V,2) == length(res), "V inconsisent with res")
#     poly_basis!(V, degree, pts, work, Val(Dim))
#     fill!(res, zero(eltype(res)))
#     for i in axes(res)        
#         for (j,w) in enumerate(wts)
#             res[i] += V[j,i]*w
#         end
#         res[i] -= moments[i]
#     end
#     return nothing
# end

# function surf_quad_jac!(jac, pts, wts, degree, V, dV, work, moments)
#     @assert( degree >= 0, "degree must be non-negative" )
#     @assert( size(pts,2) == length(wts), "pts inconsistent with wts" )
#     @assert( size(jac,1) == length(moments), "jac inconsistent with moments" )
#     Dim = size(pts,1)
#     num_quad = size(pts,2)
#     @assert( size(jac,2) == (Dim+1)*num_quad, "jac inconsistent with wts/pts" )
#     @assert( size(V,1) == size(dV,1) == num_quad, "V/dV inconsisent with pts" )
#     @assert( size(V,2) == size(dV,2) == size(jac,1) )
#     poly_basis!(V, degree, pts, work, Val(Dim))
#     poly_basis_derivatives!(dV, degree, pts, Val(Dim))
#     for i in axes(jac,1)
#         # Jacobian w.r.t. weights 
#         for j in axes(wts)
#             jac[i,j] = V[j,i]
#         end
#         # Jacobian w.r.t. quadrature nodes
#         ptr = length(wts)
#         for d = 1:Dim
#             for (j,w) in enumerate(wts)
#                 jac[i,ptr+j] = dV[j,i,d]*w
#             end
#             ptr += length(wts)
#         end
#     end
#     return nothing
# end

# function surf_quad_obj(x, degree, res, V, work, moments, Dim)
#     num_quad = size(V,1)
#     wts = view(x,1:num_quad)
#     pts = reshape(view(x,num_quad+1:num_quad*(Dim+1)), (Dim, num_quad)) 
#     surf_quad_res!(res, pts, wts, degree, V, work, moments)
#     return 0.5*dot(res, res)
# end

# function surf_quad_grad!(g, x, degree, res, jac, V, dV, work, moments, Dim)
#     num_quad = size(V,1)
#     wts = view(x,1:num_quad)
#     pts = reshape(view(x,num_quad+1:num_quad*(Dim+1)), (Dim, num_quad)) 
#     surf_quad_res!(res, pts, wts, degree, V, work, moments)
#     surf_quad_jac!(jac, pts, wts, degree, V, dV, work, moments)
#     for i = 1:length(x)
#         g[i] = dot(res, jac[:,i])
#     end
#     return nothing
# end

# function surf_quad_hess!(hess, x, degree, jac, V, dV, work, momemnts, Dim)
#     num_quad = size(V,1)
#     wts = view(x,1:num_quad)
#     pts = reshape(view(x,num_quad+1:num_quad*(Dim+1)), (Dim, num_quad)) 
#     surf_quad_jac!(jac, pts, wts, degree, V, dV, work, moments)
#     for i = 1:length(x)
#         hess[i,j] = dot(jac[:,i], jac[:,i])
#         for j = i+1:length(x)
#             hess[i,j] = dot(jac[:,i], jac[:,j])
#             hess[j,i] = hess[i,j]
#         end
#     end
#     return nothing
# end