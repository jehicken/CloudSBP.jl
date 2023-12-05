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
    obj = penalty(root, xc, xc_init, dist_ref, mu, degree)

Compute the objective that seeks to minimize the change in the node locations
while ensuring positive quadrature weights.  `root` is the mesh, `xc` are the 
nodes being varied, `xc_init` are the initial node locations, `dist_ref` are
reference lengths, `mu` is the penalty parameter for negative quad weights,
and `degree` is the target exactness of the rule.
"""
function penalty(root::Cell{Data, Dim, T, L}, xc, xc_init, dist_ref, mu, degree
                 )  where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    # compute the norm part of the penalty
    phi = 0.0
    for i = 1:num_nodes 
        dist = 0.0
        for d = 1:Dim 
            dist += (xc[d,i] - xc_init[d,i])^2
        end 
        phi += dist/(dist_ref[i]^2)
    end

    # compute the diagonal norm based on x 
    H = diagonal_norm(root, xc, degree)

    # add the penalties 
    for i = 1:num_nodes 
        H_ref = dist_ref[i]^Dim
        if real(H[i]) < H_ref
            phi += mu*(H[i]/H_ref - 1)^2
        end 
    end

    phi *= 0.5
    return phi
end

function penalty_grad!(g::AbstractVector{T}, root::Cell{Data, Dim, T, L}, 
                       xc, xc_init, dist_ref, mu, degree
                       )  where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    # need the diagonal norm for the reverse sweep 
    H = diagonal_norm(root, xc, degree)

    # start the reverse sweep 
    fill!(g, zero(T))
    # return phi
    # phi *= 0.5 
    phi_bar = 0.5 
    # add the penalties 
    H_bar = zero(H)
    for i = 1:num_nodes 
        H_ref = dist_ref[i]^Dim
        if real(H[i]) < H_ref
            # phi += mu*(H[i]/H_ref - 1)^2
            H_bar[i] += phi_bar*2.0*mu*(H[i]/H_ref - 1)/H_ref
        end 
    end

    # compute the diagonal norm based on x 
    #H = diagonal_norm(root, xc, degree)
    xc_bar = reshape(g, size(xc))
    diagonal_norm_rev!(xc_bar, root, xc, degree, H_bar)

    # compute the norm part of the penalty
    for i = 1:num_nodes 
        # phi += dist/(dist_ref[i]^2)
        dist_bar = phi_bar/(dist_ref[i]^2)
        for d = 1:Dim 
            # dist += (xc[d,i] - xc_init[d,i])^2
            xc_bar[d,i] += dist_bar*2.0*(xc[d,i] - xc_init[d,i])
        end 
    end
    return nothing
end 