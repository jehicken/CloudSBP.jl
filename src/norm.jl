# Routines related to constructing a diagonal norm

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
    m = calc_moments(root, levset, degree)

Returns the first total `degree` integral moments for all cells in the tree
defined by `root`.  The tree must have been preprocessed to identify poentially
cut and immersed cells using the `levset` level-set function.  In addition, the 
`cell.data.xref` and `cell.data.dx` fields must containt the reference origin
for each cell.

WARNING: The signature of the function `levset` must of the form
levset(Vector{Float64})::Float64, because this assumption is used when it is
wrapped using `csafe_function`.
"""
function calc_moments(root::Cell{Data, Dim, T, L}, levset, degree
                      ) where {Data, Dim, T, L}
    num_cell = num_leaves(root)
    num_basis = binomial(Dim + degree, Dim)
    moments = zeros(num_basis, num_cell)

    # get arrays/data used for tensor-product quadrature 
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(w1d)^Dim             
    wq = zeros(num_quad)
    xq = zeros(Dim, num_quad)
    Vq = zeros(num_quad, num_basis)
    workq = zeros((Dim+1)*num_quad)

    # set up the level-set function for passing to calc_cut_quad below
    mod_levset[] = levset
    #safe_clevset = @safe_cfunction( 
    #    x -> evallevelset(x, mod_levset[]), Cdouble, (Vector{Float64},))
    safe_clevset = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))

    for (c, cell) in enumerate(allleaves(root))
        xavg = cell.data.xref #view(cell_xavg, :, c)
        dx = cell.data.dx #view(cell_dx, :, c)
        if cell.data.immersed
            # do not integrate cells that have been confirmed immersed
            continue
        elseif is_cut(cell)
            # this cell *may* be cut; use Saye's algorithm
            wq_cut, xq_cut, surf_wts, surf_pts = calc_cut_quad(
                cell.boundary, safe_clevset, degree+1, fit_degree=degree)
            # consider resizing 1D arrays here, if need larger
            for I in CartesianIndices(xq_cut)
                xq_cut[I] = (xq_cut[I] - xavg[I[1]])/dx[I[1]] - 0.5
            end
            Vq_cut = zeros(length(wq_cut), num_basis)
            workq_cut = zeros((Dim+1)*length(wq_cut))
            poly_basis!(Vq_cut, degree, xq_cut, workq_cut, Val(Dim))
            for i = 1:num_basis
                moments[i, c] = dot(Vq_cut[:,i], wq_cut)
            end
            cell.data.moments = view(moments, :, c)
        else
            # this cell is not cut; use a tensor-product quadrature to integrate
            # Precompute?
            quadrature!(xq, wq, cell.boundary, x1d, w1d)
            for I in CartesianIndices(xq)
                xq[I] = (xq[I] - xavg[I[1]])/dx[I[1]] - 0.5
            end
            poly_basis!(Vq, degree, xq, workq, Val(Dim))
            for i = 1:num_basis
                moments[i, c] = dot(Vq[:,i], wq)
            end
            cell.data.moments = view(moments, :, c)
        end
    end 
    return moments
end

"""
    m = calc_moments(root, degree)

This variant is useful when you want moments for all cells in the tree `root`.
"""
function calc_moments(root::Cell{Data, Dim, T, L}, degree) where {Data, Dim, T, L}
    levset(x) = 1.0
    return calc_moments(root, levset, degree)
end

"""
    w = cell_quadrature(degree, xc, xq, wq, Val(Dim))

Given a quadrature rule `(xq,wq)` that is exact for polynomials of `degree` 
degree, computes a new quadrature for the same domain based on the nodes `xc`.
"""
function cell_quadrature(degree, xc, xq, wq, ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == size(xq,1) == Dim, "xc/xq/Dim are inconsistent")
    @assert( size(xq,2) == size(wq,1), "xq and wq have inconsistent sizes")
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = size(xc, 2)
    num_quad = size(xq, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc and xq
    lower = minimum([real.(xc) real.(xq)], dims=2)
    upper = maximum([real.(xc) real.(xq)], dims=2)
    dx = upper - lower 
    xavg = 0.5*(upper + lower)
    # xavg .*= 0.0
    dx[:] .*= 1.001
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    xq_trans = zero(xq)
    for I in CartesianIndices(xq)
        xq_trans[I] = (xq[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the quadrature and node points 
    workc = zeros(eltype(xc), (Dim+1)*num_nodes)
    V = zeros(eltype(xc), num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    workq = zeros((Dim+1)*num_quad)
    Vq = zeros(num_quad, num_basis)
    poly_basis!(Vq, degree, xq_trans, workq, Val(Dim))
    # integrate the polynomial basis using the given quadrature
    b = zeros(num_basis)
    for i = 1:num_basis
        b[i] = dot(Vq[:,i], wq)
    end
    w = zeros(num_nodes)
    solve_min_norm!(w, V, b)
    return w
end

function cell_quadrature(degree, xc::AbstractArray{ComplexF64,2}, xq, wq,
                         ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == size(xq,1) == Dim, "xc/xq/Dim are inconsistent")
    @assert( size(xq,2) == size(wq,1), "xq and wq have inconsistent sizes")
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = size(xc, 2)
    num_quad = size(xq, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc and xq
    lower = minimum([real.(xc) real.(xq)], dims=2)
    upper = maximum([real.(xc) real.(xq)], dims=2)
    dx = upper - lower 
    xavg = 0.5*(upper + lower)
    #xavg .*= 0.0
    dx[:] .*= 1.001
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    xq_trans = zero(xq)
    for I in CartesianIndices(xq)
        xq_trans[I] = (xq[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the quadrature and node points 
    workc = zeros(eltype(xc), (Dim+1)*num_nodes)
    V = zeros(eltype(xc), num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    workq = zeros((Dim+1)*num_quad)
    Vq = zeros(num_quad, num_basis)
    poly_basis!(Vq, degree, xq_trans, workq, Val(Dim))
    # integrate the polynomial basis using the given quadrature
    b = zeros(num_basis)
    for i = 1:num_basis
        b[i] = dot(Vq[:,i], wq)
    end
    w = zeros(num_nodes)
    dw = zeros(num_nodes)
    dV = imag.(V)
    solve_min_norm_diff!(w, dw, real.(V), dV, b)    
    return complex.(w, dw)
end

"""
    w = cell_quadrature(degree, xc, moments, xavg, dx, Val(Dim))

Given a set of total `degree` polynomial `moments`, computes a quadrature that
is exact for those moments based on the nodes `xc`.  The arrays `xavg` and `dx`
are used to shift and scale, respectively, the nodes in `xc` to improve 
conditioning of the Vandermonde matrix.  The same scaling and shifts must have
been applied when computing the integral `moments`.
"""
function cell_quadrature(degree, xc, moments, xavg, dx, ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == length(dx) == length(xavg) == Dim,
             "xc/dx/xavg/Dim are inconsistent")
    num_basis = binomial(Dim + degree, Dim)
    @assert( length(moments) == num_basis, "moments has inconsistent size")
    num_nodes = size(xc, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the node points 
    workc = zeros(eltype(xc), (Dim+1)*num_nodes)
    V = zeros(eltype(xc), num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    # find the weights that satisfy the moments
    w = zeros(num_nodes)
    solve_min_norm!(w, V, moments)
    return w
end

function cell_quadrature(degree, xc::AbstractArray{ComplexF64,2}, moments, xavg,
                         dx, ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == length(dx) == length(xavg) == Dim,
             "xc/dx/xavg/Dim are inconsistent")
    num_basis = binomial(Dim + degree, Dim)
    @assert( length(moments) == num_basis, "moments has inconsistent size")
    num_nodes = size(xc, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the node points 
    workc = zeros(eltype(xc), (Dim+1)*num_nodes)
    V = zeros(eltype(xc), num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    # find the weights that satisfy the moments
    w = zeros(num_nodes)
    dw = zeros(num_nodes)
    dV = imag.(V)
    solve_min_norm_diff!(w, dw, real.(V), dV, moments)    
    return complex.(w, dw)
end

"""
    cell_quadrature_rev!(xc_bar, degree, xc, xq, wq, w_bar, Val(Dim))

Reverse mode differentiated `cell_quadrature`.  Returns the derivatives of 
`dot(w, w_bar)` with respect to `xc` in the array `xc_bar`.  All other inputs
are the same as `cell_quadrature`.
"""
function cell_quadrature_rev!(xc_bar, degree, xc, xq, wq, w_bar, ::Val{Dim}
                              ) where {Dim}
    @assert( size(xc,1) == size(xq,1) == Dim, "xc/xq/Dim are inconsistent")
    @assert( size(xq,2) == size(wq,1), "xq and wq have inconsistent sizes")
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = size(xc, 2)
    num_quad = size(xq, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc and xq
    lower = minimum([real.(xc) real.(xq)], dims=2)
    upper = maximum([real.(xc) real.(xq)], dims=2)
    dx = upper - lower 
    xavg = 0.5*(upper + lower)
    #xavg .*= 0.0
    dx[:] .*= 1.001
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    xq_trans = zero(xq)
    for I in CartesianIndices(xq)
        xq_trans[I] = (xq[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the quadrature and node points 
    workc = zeros((Dim+1)*num_nodes)
    V = zeros(num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    dV = zeros(num_nodes, num_basis, Dim)
    poly_basis_derivatives!(dV, degree, xc_trans, Val(Dim))
    workq = zeros((Dim+1)*num_quad)
    Vq = zeros(num_quad, num_basis)
    poly_basis!(Vq, degree, xq_trans, workq, Val(Dim))
    # integrate the polynomial basis using the given quadrature
    b = zeros(num_basis)
    for i = 1:num_basis
        b[i] = dot(Vq[:,i], wq)
    end
    # compute the derivative of the objective w.r.t. V
    V_bar = zero(V)
    solve_min_norm_rev!(V_bar, w_bar, V, b)
    # compute the derivative of the objective w.r.t. xc 
    for d = 1:Dim
        xc_bar[d,:] += sum(V_bar[:,:].*dV[:,:,d],dims=2)/dx[d]
    end
    return nothing
end

"""
    cell_quadrature_rev!(xc_bar, degree, xc, moments, xavg, dx, w_bar, Val(Dim))

Reverse mode differentiated of the moment-based variant of `cell_quadrature`.
Returns the derivatives of `dot(w, w_bar)` with respect to `xc` in the array
`xc_bar`.  All other inputs are the same as the moment-based variant of
`cell_quadrature`.
"""
function cell_quadrature_rev!(xc_bar, degree, xc, moments, xavg, dx, w_bar,
                              ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == size(xc_bar,1) == length(dx) == length(xavg) == Dim,
             "xc/xc_bar/dx/xavg/Dim are inconsistent")
    num_basis = binomial(Dim + degree, Dim)
    @assert( length(moments) == num_basis, "moments has inconsistent size")
    num_nodes = size(xc, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the node points 
    workc = zeros(eltype(xc), (Dim+1)*num_nodes)
    V = zeros(eltype(xc), num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    dV = zeros(num_nodes, num_basis, Dim)
    poly_basis_derivatives!(dV, degree, xc_trans, Val(Dim))
    # compute the derivative of the objective w.r.t. V
    V_bar = zero(V)
    solve_min_norm_rev!(V_bar, w_bar, V, moments)
    # compute the derivative of the objective w.r.t. xc 
    for d = 1:Dim
        xc_bar[d,:] += sum(V_bar[:,:].*dV[:,:,d],dims=2)/dx[d]
    end
    return nothing
end

function diagonal_norm(root::Cell{Data, Dim, T, L}, points, degree
                       ) where {Data, Dim, T, L}
    num_nodes = size(points, 2)
    H = zeros(eltype(points), num_nodes)
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes                   
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    for cell in allleaves(root)
        # get the nodes in this cell's stencil, and an accurate quaduature
        #nodes = copy(points[:, cell.data.points])
        nodes = view(points, :, cell.data.points)
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        # get cell quadrature and add to global norm
        w = cell_quadrature(degree, nodes, xq, wq, Val(Dim))
        #w = nodes[Dim,:].*nodes[1,:]
        for i = 1:length(cell.data.points)
            H[cell.data.points[i]] += w[i]
        end
    end
    return H
end

function diagonal_norm(root::Cell{Data, Dim, T, L}, points, degree
                       ) where {Data, Dim, T, L}
    num_nodes = size(points, 2)
    @assert( num_nodes == size(moments,2), "points/moments inconsistent" )
    @assert( size(points,1) == size(xavg,1) == size(dx,1) == Dim,
            "points/xavg/dx inconsistent with Dim" )
    H = zeros(eltype(points), num_nodes)
    for cell in allleaves(root)
        # get the nodes in this cell's stencil
        nodes = view(points, :, cell.data.points)
        # get cell quadrature and add to global norm
        w = cell_quadrature(degree, nodes, cell.data.moments, cell.data.xref,
                            cell.data.dx, Val(Dim))
        for i = 1:length(cell.data.points)
            H[cell.data.points[i]] += w[i]
        end
    end
    return H
end

function diagonal_norm_rev!(points_bar, root::Cell{Data, Dim, T, L}, points, 
                            degree, H_bar) where {Data, Dim, T, L}
    num_nodes = size(points, 2)
    fill!(points_bar, zero(T))    
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes                   
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    for cell in allleaves(root)
        # get the nodes in this cell's stencil, and an accurate quaduature
        #nodes = copy(points[:, cell.data.points])
        nodes = view(points, :, cell.data.points)
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        w_bar = zeros(length(cell.data.points))
        for i = 1:length(cell.data.points)
            # H[cell.data.points[i]] += w[i]
            w_bar[i] = H_bar[cell.data.points[i]]
        end
        nodes_bar = view(points_bar, :, cell.data.points)
        # w = cell_quadrature(degree, nodes, xq, wq, Val(Dim))
        cell_quadrature_rev!(nodes_bar, degree, nodes, xq, wq, w_bar, Val(Dim))
        #nodes_bar[Dim,:] += w_bar[:].*nodes[1,:]
        #nodes_bar[1,:] += w_bar[:].*nodes[Dim,:]
    end
    return nothing
end

"""
    Z, wp = cell_null_and_part(degree, xc, xq, wq, Val(Dim))

Given a quadrature rule `(xq,wq)` that is exact for polynomials of `degree` 
degree, computes a new quadrature for the same domain based on the nodes `xc`.  
This new rule is returned as `wp`.  Also returns the nullspace for the 
quadrature moment matching problem.
"""
function cell_null_and_part(degree, xc, xq, wq, ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == size(xq,1) == Dim, "xc/xq/Dim are inconsistent")
    @assert( size(xq,2) == size(wq,1), "xq and wq have inconsistent sizes")
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = size(xc, 2)
    num_quad = size(xq, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc and xq
    lower = minimum([real.(xc) real.(xq)], dims=2)
    upper = maximum([real.(xc) real.(xq)], dims=2)
    dx = upper - lower 
    xavg = 0.5*(upper + lower)
    xavg .*= 0.0
    dx[:] .*= 1.001
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    xq_trans = zero(xq)
    for I in CartesianIndices(xq)
        xq_trans[I] = (xq[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the quadrature and node points 
    workc = zeros((Dim+1)*num_nodes)
    V = zeros(num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    workq = zeros((Dim+1)*num_quad)
    Vq = zeros(num_quad, num_basis)
    poly_basis!(Vq, degree, xq_trans, workq, Val(Dim))
    # integrate the polynomial basis using the given quadrature
    b = zeros(num_basis)
    for i = 1:num_basis
        b[i] = dot(Vq[:,i], wq)
    end
    wp = V'\b 
    Z = nullspace(V')
    return Z, wp 
end

"""
    Z, wp, num_var = get_null_and_part(root, points, degree)

Returns a vector of nullspaces, `Z`, and particular solutions, `wp`, for each 
cell's quadrature problem.  Also returns the total number of degrees of freedom.
"""
function get_null_and_part(root::Cell{Data, Dim, T, L}, points, degree
                           ) where {Data, Dim, T, L}
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes                   
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))                             
    num_cell = num_leaves(root)
    Z = Vector{Matrix{T}}(undef, num_cell)
    wp = Vector{Vector{T}}(undef, num_cell)
    num_var = 0
    for (c, cell) in enumerate(allleaves(root))
        # get the nodes in this cell's stencil, and an accurate quaduature
        nodes = copy(points[:, cell.data.points])
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        # get cell quadrature and add to global norm
        Z[c], wp[c]  = cell_null_and_part(degree, nodes, xq, wq, Val(Dim))
        num_var += size(Z[c], 2)
    end
    return Z, wp, num_var
end

"""
    obj = obj_norm(root, wp, Z, y, rho, num_nodes)

Compute the objective to maximize the minimum norm.  `root` is the mesh, which 
is mostly needed to known the stencil for each cell.  `wp` is a vector of 
vectors holding the particular solution for each cell, and `Z` is the nullspace 
for each cell's problem.  
"""
function obj_norm(root::Cell{Data, Dim, T, L}, wp::AbstractVector{Vector{T2}},
                  Z::AbstractVector{Matrix{T2}}, y::AbstractVector{T2},
                  rho::T2, num_nodes::Int) where {Data, Dim, T, L, T2}
    # reconstruct the norm from the particular solution and the null space
    H = zeros(T2, num_nodes)
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        num_dof = size(Z[c],2)
        w = wp[c] + Z[c]*y[ptr+1:ptr+num_dof]
        ptr += num_dof
        for i = 1:length(cell.data.points)            
            H[cell.data.points[i]] += w[i]
        end
    end
    # compute the discrete KS function
    minH = minimum(real.(H)) 
    obj = -minH
    sum_exp = 0.0
    for i = 1:num_nodes 
        sum_exp += exp(rho*(minH - H[i]))
    end 
    obj += log(sum_exp/num_nodes)/rho
    return obj 
end

"""
    obj_norm_grad!(g, root, wp, Z, y, num_nodes)

Compute the gradient of the objective to maximize the minimum norm.  `root` is 
the mesh, which is mostly needed to known the stencil for each cell.  `wp` is a 
vector of vectors holding the particular solution for each cell, and `Z` is the 
nullspace for each cell's problem.  
"""
function obj_norm_grad!(g::AbstractVector{T}, root::Cell{Data, Dim, T, L}, 
                        wp::AbstractVector{Vector{T}},
                        Z::AbstractVector{Matrix{T}}, y::AbstractVector{T},
                        rho::T, num_nodes::Int) where {Data, Dim, T, L}
    # forward sweep
    H = zeros(T, num_nodes)
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        num_dof = size(Z[c],2)
        w = wp[c] + Z[c]*y[ptr+1:ptr+num_dof]
        ptr += num_dof
        for i = 1:length(cell.data.points)            
            H[cell.data.points[i]] += w[i]
        end
    end
    # compute the discrete KS function
    minH = minimum(H) 
    obj = -minH
    sum_exp = 0.0
    for i = 1:num_nodes 
        sum_exp += exp(rho*(minH - H[i]))
    end 
    obj += log(sum_exp/num_nodes)/rho
    #return obj 

    # reverse sweep
    obj_bar = 1.0
    #obj += ln(sum_exp/num_nodes)/rho
    sum_exp_bar = obj_bar/(sum_exp * rho)
    H_bar = zeros(T, num_nodes)
    for i = 1:num_nodes 
        # sum_exp += exp(rho*(minH - H[i]))
        H_bar[i] -= sum_exp_bar*exp(rho*(minH - H[i]))*rho
    end 
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        w_bar = zero(wp[c])
        for i = 1:length(cell.data.points)            
            # H[cell.data.points[i]] += w[i]
            w_bar[i] += H_bar[cell.data.points[i]]
        end
        # w = wp[c] + Z[c]*y[ptr+1:ptr+num_dof]
        num_dof = size(Z[c],2)
        g[ptr+1:ptr+num_dof] += Z[c]'*w_bar
        ptr += num_dof
    end
    return nothing
end 

"""
    obj = penalty(root, xc, xc_init, dist_ref, H_tol, mu, degree)

Compute the objective that seeks to minimize the change in the node locations
while ensuring positive quadrature weights.  `root` is the mesh, `xc` are the 
nodes being varied, and `xc_init` are the initial node locations. `dist_ref` are
reference lengths, and `H_tol` is an array of tolerances for the quadrature 
weight at each node; that is, `H[i] >= H_tol[i]` for the constraint to be 
satisfied.  Finally, `mu` scales the regularization term, and `degree` is the 
target exactness of the rule.
"""
function penalty(root::Cell{Data, Dim, T, L}, xc, xc_init, dist_ref, H_tol, 
                 mu, degree)  where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    # compute the norm part of the penalty
    phi = 0.0
    for i = 1:num_nodes 
        dist = 0.0
        for d = 1:Dim 
            dist += (xc[d,i] - xc_init[d,i])^2
        end 
        phi += 0.5*mu*dist/(dist_ref[i]^2)
    end

    # compute the diagonal norm based on x 
    H = diagonal_norm(root, xc, degree)

    # # compute the discrete KS function
    # rho = 100.0
    # minH = minimum(real.(H))     
    # sum_exp = 0.0
    # for i = 1:num_nodes 
    #     sum_exp += exp(rho*(minH - H[i]))
    # end 
    # phi += -minH + log(sum_exp/num_nodes)/rho

    # add the penalties
    for i = 1:num_nodes
        if real(H[i]) < H_tol[i]
            phi += 0.5*(H[i]/H_tol[i] - 1)^2
        end 
    end

    return phi
end



function penalty_grad!(g::AbstractVector{T}, root::Cell{Data, Dim, T, L}, 
                       xc, xc_init, dist_ref, H_tol, mu, degree
                       )  where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    # need the diagonal norm for the reverse sweep 
    H = diagonal_norm(root, xc, degree)

    # rho = 100.0
    # minH = minimum(real.(H))
    # sum_exp = 0.0
    # for i = 1:num_nodes 
    #     sum_exp += exp(rho*(minH - H[i]))
    # end 

    # start the reverse sweep 
    fill!(g, zero(T))

    # return phi
    # phi *= 0.5 
    # phi_bar = 0.5 

    # return phi 
    phi_bar = 1.0 

    # # phi += -minH + log(sum_exp/num_nodes)/rho
    # sum_exp_bar = phi_bar/(sum_exp * rho)
    # H_bar = zero(H)
    # for i = 1:num_nodes 
    #     # sum_exp += exp(rho*(minH - H[i]))
    #     H_bar[i] -= sum_exp_bar*exp(rho*(minH - H[i]))*rho
    # end

    # add the penalties    
    H_bar = zero(H)
    for i = 1:num_nodes         
        if real(H[i]) < H_tol[i] 
            # phi += 0.5*(H[i]/H_tol[i] - 1)^2
            H_bar[i] += phi_bar*(H[i]/H_tol[i] - 1)/H_tol[i]
        end 
    end

    # compute the diagonal norm based on x 
    #H = diagonal_norm(root, xc, degree)
    xc_bar = reshape(g, size(xc))
    diagonal_norm_rev!(xc_bar, root, xc, degree, H_bar)

    # compute the norm part of the penalty
    for i = 1:num_nodes 
        # phi += 0.5*mu*dist/(dist_ref[i]^2)
        dist_bar = 0.5*mu*phi_bar/(dist_ref[i]^2)
        for d = 1:Dim 
            # dist += (xc[d,i] - xc_init[d,i])^2
            xc_bar[d,i] += dist_bar*2.0*(xc[d,i] - xc_init[d,i])
        end 
    end
    return nothing
end 

function penalty_block_hess!(p::AbstractVector{T}, g::AbstractVector{T},
                             root::Cell{Data, Dim, T, L}, xc, dist_ref,
                             mu, degree)  where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    dxc = reshape(p, Dim, num_nodes)
    gc = reshape(g, Dim, num_nodes)
    # need the diagonal norm
    H = diagonal_norm(root, xc, degree)

    dHdx = zeros(2, num_nodes) 
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes                   
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    for cell in allleaves(root)
        # get the nodes in this cell's stencil, and an accurate quaduature
        #nodes = copy(points[:, cell.data.points])
        nodes = view(xc, :, cell.data.points)
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        w_bar = zeros(length(cell.data.points))
        dwdx = zeros(Dim, length(cell.data.points))
        for i = 1:length(cell.data.points)            
            fill!(dwdx, 0.0)
            w_bar[i] = 1.0 
            cell_quadrature_rev!(dwdx, degree, nodes, xq, wq, w_bar, Val(Dim))
            dHdx[:,cell.data.points[i]] += dwdx[:,i]
            w_bar[i] = 0.0
        end
    end
    # now form the (approximate) Dim x Dim diagonal block of the Hessian, and
    # invert it on the given vector.
    hess = zeros(Dim, Dim)
    tol = 1e-5
    for i = 1:num_nodes 
        #dist = 0.0
        #for d = 1:Dim 
        #    dist += (xc[d,i] - xc_init[d,i])^2
        #end 
        #phi += 0.5*mu*dist/(dist_ref[i]^2)
        hess[:,:] = (mu/dist_ref[i]^2)*diagm(ones(Dim))

        # H_ref = dist_ref[i]^Dim
        H_ref = 1.0
        if real(H[i]) < tol #H_ref
            # phi += 0.5*(H[i]/tol - 1)^2
            hess[:,:] += dHdx[:,i]*(dHdx[:,i]'/tol^2)
        end

        dxc[:,i] = hess\gc[:,i]
        if dot(dxc[:,i], gc[:,i]) < 0.0
            # dxc is not (locally) a descent direction, so revert 
            dxc[:,i] = (dist_ref[i]^2/mu)*gc[:,i]
        end
    end
    return nothing
end

function apply_approx_inverse!(p::AbstractVector{T}, g::AbstractVector{T},
                               root::Cell{Data, Dim, T, L}, xc, dist_ref, H_tol,
                               mu, degree, max_rank)  where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    # need the diagonal norm for various steps
    H = diagonal_norm(root, xc, degree)

    # determine the set of norms that are smallest
    tol = 1e-5
    small_H = Vector{Int}()
    diag_scal = Vector{T}()
    for i = 1:num_nodes
        if real(H[i]) < H_tol[i]
            push!(small_H, i)
            #push!(diag_scal, 1/H_tol[i])
            push!(diag_scal, real(H[i]))
        end
    end
    println("Number of violating weights = ",length(small_H), "/", num_nodes)
    #println("small_H = ",H[small_H])
    if length(small_H) > max_rank
        # truncate at maximum allowable rank
        sort_idx = sortperm(diag_scal)
        small_H = small_H[sort_idx[1:max_rank]]
    end 
    # done with diag_scal's temporary use, so store actual scales
    diag_scal = 1.0./H_tol[small_H]

    #println("After trucation...")
    #println("small_H = ",H[small_H])

    # construct the Jacobian for the smallest norms 
    H_Jac = zeros(Dim, num_nodes, length(small_H))
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes                   
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    for cell in allleaves(root)
        # first, check if any of the nodes in this cell's stencil are among the 
        # small_H indices 
        cell_set = intersect(small_H, cell.data.points)
        if length(cell_set) == 0
            continue
        end
        #println("cell.data.points = ",cell.data.points)
        #println("cell_set = ",cell_set)
        # get the nodes in this cell's stencil, and an accurate quaduature
        nodes = view(xc, :, cell.data.points)
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        # loop over the nodes in cell_set 
        w_bar = zeros(length(cell.data.points))
        for i = 1:length(cell_set)
            idx = findfirst(j -> j == cell_set[i], cell.data.points)
            w_bar[idx] = 1.0
            nodes_bar = view(H_Jac, :, cell.data.points, findfirst(j -> j == cell_set[i], small_H)) #, Dim, length(cell.data.points))
            cell_quadrature_rev!(nodes_bar, degree, nodes, xq, wq, w_bar,
                                 Val(Dim))
            w_bar[idx] = 0.0
        end
    end

    # Compute the QR factorization of H_Jac; could do this in place to save 
    # memory
    A = reshape(H_Jac, Dim*num_nodes, length(small_H))
    A *= diagm(diag_scal)
    F = qr(A)

    # find the search direction 
    Q = Matrix(F.Q)
    p[:] = -Q *( (F.R*F.R')\(Q'*g) )

    return nothing
end