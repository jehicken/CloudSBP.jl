# Routines related to constructing a diagonal norm


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

    # find the scaling factors 
    dx = 1e16*ones(num_nodes)
    for i = 1:num_nodes
        for j = 1:num_nodes 
            if i == j 
                continue 
            end
            dist = norm(xc_trans[:,j] - xc_trans[:,i])
            dist < dx[i] ? dx[i] = dist : nothing 
        end 
    end
    dx = dx.^Dim 
    #dx = sqrt.(dx)
    #dx = 1.0./dx
    #dx = dx.^2

    A = V'*diagm(dx)
    w = A\b 
    return diagm(dx)*w 

    #w = pinv(V')*b
    #w = V'\b
    #R = diagm(ones(num_nodes).*(prod(dx)/num_nodes))
    #w = R*V*((V'*R*V)\b)
    #@assert( norm(V'*w - b) < (1e-15)*(10^degree),
    #         "quadrature is not accurate!" )
    #if norm(V'*w - b) > (1e-15)*(10^degree)
    #    println("WARNING: quadrature is not accurate! res = ", norm(V'*w - b))
    #end
    #return w
end

function diagonal_norm(root::Cell{Data, Dim, T, L}, points, degree
                       ) where {Data, Dim, T, L}
    num_nodes = size(points, 2)
    H = zeros(T, num_nodes)
    # find the maximum number of phi basis over all cells
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes                   
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    for cell in allleaves(root)
        # get the nodes in this cell's stencil, and an accurate quaduature
        nodes = copy(points[:, cell.data.points])
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        # get cell quadrature and add to global norm
        w = cell_quadrature(degree, nodes, xq, wq, Val(Dim))
        for i = 1:length(cell.data.points)
            H[cell.data.points[i]] += w[i]
        end
    end
    return H
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
