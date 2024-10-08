# Methods related to optimizing the node locations for the norm inequality

"""
    w = cell_quadrature(degree, xc, moments, xref, dx, Val(Dim))

Partially complexified version of `cell_quadrature`; partially, because 
`solve_min_norm!` does not work with complex variables, so that function is 
differentiated explicitly here.
"""
function cell_quadrature(degree, xc::AbstractArray{ComplexF64,2}, moments, xref,
                         dx, ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == length(dx) == length(xref) == Dim,
             "xc/dx/xref/Dim are inconsistent")
    num_basis = binomial(Dim + degree, Dim)
    @assert( length(moments) >= num_basis, "moments has inconsistent size")
    num_nodes = size(xc, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xref[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the node points 
    workc = zeros(eltype(xc), (Dim+1)*num_nodes)
    V = zeros(eltype(xc), num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    # find the weights that satisfy the moments
    w = zeros(num_nodes)
    dw = zeros(num_nodes)
    dV = imag.(V)
    solve_min_norm_diff!(w, dw, real.(V), dV, moments[1:num_basis])    
    return complex.(w, dw)
end

"""
    cell_quadrature_rev!(xc_bar, degree, xc, moments, xref, dx, w_bar, Val(Dim))

Reverse mode differentiated of the moment-based variant of `cell_quadrature`.
Returns the derivatives of `dot(w, w_bar)` with respect to `xc` in the array
`xc_bar`.  All other inputs are the same as the moment-based variant of
`cell_quadrature`.
"""
function cell_quadrature_rev!(xc_bar, degree, xc, moments, xref, dx, w_bar,
                              ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == size(xc_bar,1) == length(dx) == length(xref) == Dim,
             "xc/xc_bar/dx/xref/Dim are inconsistent")
    num_basis = binomial(Dim + degree, Dim)
    @assert( length(moments) >= num_basis, "moments has inconsistent size")
    num_nodes = size(xc, 2)
    @assert( num_nodes >= num_basis, "fewer nodes than basis functions")
    # apply an affine transformation to the points xc
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xref[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the polynomial basis at the node points 
    workc = zeros(eltype(xc), (Dim+1)*num_nodes)
    V = zeros(eltype(xc), num_nodes, num_basis)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    dV = zeros(num_nodes, num_basis, Dim)
    poly_basis_derivatives!(dV, degree, xc_trans, Val(Dim))
    # compute the derivative of the objective w.r.t. V
    V_bar = zero(V)
    solve_min_norm_rev!(V_bar, w_bar, V, moments[1:num_basis])
    # compute the derivative of the objective w.r.t. xc 
    for d = 1:Dim
        xc_bar[d,:] += sum(V_bar[:,:].*dV[:,:,d],dims=2)/dx[d]
    end
    return nothing
end

"""
    w = cell_quadrature(degree, xc, xq, wq, Val(Dim))

Complexified version of `cell_quadrature` for testing derivatives.
"""
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
    xref = 0.5*(upper + lower)
    #xref .*= 0.0
    dx[:] .*= 1.001
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xref[I[1]])/dx[I[1]] - 0.5
    end
    xq_trans = zero(xq)
    for I in CartesianIndices(xq)
        xq_trans[I] = (xq[I] - xref[I[1]])/dx[I[1]] - 0.5
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
    diagonal_norm_rev!(points_bar, H_bar, root, points, degree)

Reverse-mode differentiation of the weighted norm, `dot(H, H_bar)` with respect
to the node coordinates in points.

**PRE**: The cells in `root` must have their `data.moments`, `data.xref`, and
`data.dx` fields set.
"""
function diagonal_norm_rev!(points_bar, H_bar, root::Cell{Data, Dim, T, L},
                            points, degree) where {Data, Dim, T, L}
    @assert( size(points,1) == size(points_bar,1) == Dim,
            "points/points_bar inconsistent with Dim" )
    @assert( size(points,2) == length(H_bar), "points inconsistent with H_bar")
    fill!(points_bar, zero(T))
    w_bar = Vector{Float64}()
    for cell in allleaves(root)
        if is_immersed(cell)
            continue
        end
        # get the nodes in this cell's stencil
        nodes = view(points, :, cell.data.points)
        resize!(w_bar, length(cell.data.points))
        for i = 1:length(cell.data.points)            
            # H[cell.data.points[i]] += w[i]
            w_bar[i] = H_bar[cell.data.points[i]]
        end
        # get cell quadrature and add to global norm
        # w = cell_quadrature(degree, nodes, cell.data.moments, cell.data.xref,
        #                     cell.data.dx, Val(Dim))
        nodes_bar = view(points_bar, :, cell.data.points)
        cell_quadrature_rev!(nodes_bar, degree, nodes, cell.data.moments,
                             cell.data.xref, cell.data.dx, w_bar, Val(Dim))        
    end
    return nothing
end

"""
    obj = obj_norm(root, wp, Z, y, rho, num_nodes)

Compute the objective to maximize the minimum norm.  `root` is the mesh, which 
is mostly needed to known the stencil for each cell.  `wp` is a vector of 
vectors holding the particular solution for each cell, and `Z` is the nullspace 
for each cell's problem.

**NOTE**: this objective is deprecated.
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

function smin(x)
    delta=0.1
    return (-x + sqrt(x^2 + delta^2))/2
end

function smin_bar(smin_bar, x)
    delta=0.1
    fac = sqrt(x^2 + delta^2)
    return smin_bar*(-1.0 + x/fac)/2
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
    # compute the diagonal norm based on x...
    H = zeros(eltype(xc), num_nodes)
    diagonal_norm!(H, root, xc, degree)
    # and add the penalties
    for i = 1:num_nodes
        #phi += 0.5*smin(H[i]/H_tol[i] - 1)^2
        if real(H[i]) < H_tol[i]
            phi += 0.5*(H[i]/H_tol[i] - 1)^2
        end 
    end
    return phi
end

"""
    penalty_grad!(g, root, xc, xc_init, dist_ref, H_tol, mu, degree)

Computes the derivative of `penalty` with respect to `xc`.  See `penalty` for
an explanation of the remaining parameters.
"""
function penalty_grad!(g::AbstractVector{T}, root::Cell{Data, Dim, T, L}, 
                       xc, xc_init, dist_ref, H_tol, mu, degree
                       )  where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    # need the diagonal norm for the reverse sweep 
    H = zeros(eltype(xc), num_nodes)
    diagonal_norm!(H, root, xc, degree)
    # start the reverse sweep 
    fill!(g, zero(T))

    # return phi 
    phi_bar = 1.0 

    # add the penalties    
    H_bar = zero(H)
    for i = 1:num_nodes
        # phi += 0.5*smin(H[i]/H_tol[i] - 1)^2
        #H_bar[i] += smin_bar(phi_bar*smin(H[i]/H_tol[i]-1), H[i]/H_tol[i] - 1)/#H_tol[i]
        if real(H[i]) < H_tol[i] 
            # phi += 0.5*(H[i]/H_tol[i] - 1)^2
            H_bar[i] += phi_bar*(H[i]/H_tol[i] - 1)/H_tol[i]
        end 
    end

    # compute the diagonal norm based on x 
    #H = diagonal_norm(root, xc, degree)
    xc_bar = reshape(g, size(xc))
    diagonal_norm_rev!(xc_bar, H_bar, root, xc, degree)

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

"""
    apply_approx_inverse!(p, g, root, xc, dist_ref, H_tol, mu, degree, max_rank)

Finds a low-rank approximation to the inverse Hessian of the objective
`penalty`, and applies this to `-g` to get a search direction `p`.  The
background mesh is `root`, `xc` are the nodes/points, `dist_ref` is an array of
reference distances for each node, `mu` is the regularization parameter,
`degree` is the target degree.  The rank of the approximation is controlled by 
`max_rank`.
"""
function apply_approx_inverse!(p::AbstractVector{T}, g::AbstractVector{T},
                               root::Cell{Data, Dim, T, L}, xc, dist_ref, H_tol,
                               mu, degree, max_rank)  where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    # need the diagonal norm for various steps
    H = zeros(eltype(xc), num_nodes)
    diagonal_norm!(H, root, xc, degree)

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
    println("\tNumber of violating weights = ",length(small_H), "/", num_nodes)
    #println("small_H = ",H[small_H])
    if length(small_H) > max_rank
        # truncate at maximum allowable rank
        sort_idx = sortperm(diag_scal)
        small_H = small_H[sort_idx[1:max_rank]]
        #small_H = shuffle(small_H)[1:max_rank]
    end 
    # done with diag_scal's temporary use, so store actual scales
    diag_scal = 1.0./H_tol[small_H]

    #println("After trucation...")
    #println("small_H = ",H[small_H])

    # construct the Jacobian for the smallest norms 
    H_Jac = zeros(Dim, num_nodes, length(small_H))
    for cell in allleaves(root)
        if is_immersed(cell)
            continue
        end
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
        # loop over the nodes in cell_set 
        w_bar = zeros(length(cell.data.points))
        for i = 1:length(cell_set)
            idx = findfirst(j -> j == cell_set[i], cell.data.points)
            w_bar[idx] = 1.0
            nodes_bar = view(H_Jac, :, cell.data.points, findfirst(j -> j == cell_set[i], small_H)) #, Dim, length(cell.data.points))
            cell_quadrature_rev!(nodes_bar, degree, nodes, cell.data.moments, cell.data.xref,
                                 cell.data.dx, w_bar, Val(Dim))
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
    
    # for directions perpendicular to Q
    p[:] -= 0.001*(g - Q*(Q'*g))

    return nothing
end

"""
    H = opt_norm!(root, xc, degree, H_tol, mu, dist_ref, max_rank)

Finds an approximate minimizer for the objective `penalty` with respect to the
node coordinates `xc` and based on the background mesh `root`.  Once an
approximate solution is found, the corresponding `H` norm is returned; note 
that the `xc` coordinates are altered in the process.  The parameter `max_rank`
determines the rank of the approximate Jacobian of the norm with respective to 
the nodes, and this approximate Jacobian is used to form an approximate inverse
Hessian.  See `penalty` for explanations of the other parameters.
"""
function opt_norm!(root::Cell{Data, Dim, T, L}, xc, degree, H_tol, mu, dist_ref,
                   max_rank; hist_file::Union{String,Nothing}=nothing,
                   verbose::Bool=false
                   ) where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    xc_init = copy(xc)
    max_iter = 1000
    max_line = 10
    H = zeros(num_nodes)
    g = zeros(num_nodes*Dim)
    p = zero(g) 
    if hist_file !== nothing 
       f = open(hist_file, "w")
       println(f,"# Norm optimization history file")
       println(f,"# degree = $degree")
       println(f,"# mu = $mu")
       println(f,"# max_rank = $max_rank")
       println(f,"#")
       println(f,"# iter  obj  norm(g)  min(H)")
    end
    iter = 0
    for d = 1:degree
        if verbose
            println(repeat("*",80))
            println("Starting optimization with degree = ",d)
        end
        #if d > 1
        #    H_tol[:] = 0.5*H[:]
        #end
        for n = 1:max_iter
            iter += 1
            obj = CloudSBP.penalty(root, xc, xc_init, dist_ref, H_tol, mu, d)
            CloudSBP.penalty_grad!(g, root, xc, xc_init, dist_ref, H_tol, mu, d)
            CloudSBP.diagonal_norm!(H, root, xc, d)

            #minH = minimum(H)
            min_idx = argmin(H)
            if hist_file !== nothing
                println(f, "$iter  $obj  $(norm(g))  $(H[min_idx])")
            end
            if verbose
                println("\titer ",iter,": obj = ",obj,": norm(grad) = ",norm(g),
                        ": min H = ",H[min_idx])
            end
            if H[min_idx] > 0.9*H_tol[min_idx]
                break
            end
            
            CloudSBP.apply_approx_inverse!(p, g, root, xc, dist_ref, H_tol,
                                         mu, d, max_rank)
    
            alpha = 1.0 
            dxc = reshape(p, (Dim, num_nodes))
            obj0 = obj
            for k = 1:max_line
                xc[:,:] += alpha*dxc
                obj = CloudSBP.penalty(root, xc, xc_init, dist_ref, H_tol, mu, d)
                if verbose 
                    println("\t\tline-search iter ",k,": alpha = ",alpha,
                            ": obj0 = ",obj0,": obj = ",obj)
                end
                if obj < obj0
                    break
                end
                xc[:,:] -= alpha*dxc
                alpha *= 0.1
            end
        end
    end # degree loop
    if hist_file !== nothing 
        close(f)
    end
    return H
end

"""
    obj_slice(root, xc, degree, H_tol, mu, dist_ref, node, pert)

Returns an array of objective function values along a slice based on perturbing 
node `node` in the direction `pert`.  Used for visualizing.
"""
function obj_slice(root::Cell{Data, Dim, T, L}, xc, degree, H_tol, mu, dist_ref,
                   node, pert) where {Data, Dim, T, L}
    xc_pert = copy(xc)
    obj = zeros(size(pert,2))
    for j in axes(pert,2)
        xc_pert[:,node] = xc[:,node] + pert[:,j]
        obj[j] = CloudSBP.penalty(root, xc_pert, xc, dist_ref, H_tol, mu, degree)
    end
    return obj
end 