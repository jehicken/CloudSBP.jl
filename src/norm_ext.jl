# This file is not currently used, but it is kept in the repository in case the 
# methods are needed in the future.

"""
    diagonal_norm!(H, root, wp, Z, y)

Constructs a diagonal norm (i.e. quadrature) based on the background mesh
`root`, the particular cell quaratures in `wp`, and the null-space vectors in
`Z` and the null-space coefficients in `y`.
"""
function diagonal_norm!(H, root::Cell{Data, Dim, T, L}, wp, Z, y
                        ) where {Data, Dim, T, L}
    @assert( size(wp) == size(Z), "wp and Z are inconsistent")
    fill!(H, zero(eltype(H)))
    ptr = 0
    w = Vector{Float64}()
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        # compute the cell quadrature rule
        num_dof = size(Z[c],2)
        resize!(w, length(cell.data.points))
        w = wp[c] + Z[c]*y[ptr+1:ptr+num_dof]
        ptr += num_dof
        for i = 1:length(cell.data.points)            
            H[cell.data.points[i]] += w[i]
        end
   end
   return H
end

"""
    diagonal_norm_rev!(y_bar, H_bar, root, Z)

Reverse-mode differentiation of the weighted norm, `dot(H, H_bar)` with respect
to the null-space coefficients.
"""
function diagonal_norm_rev!(y_bar, H_bar, root::Cell{Data, Dim, T, L}, Z
                            ) where {Data, Dim, T, L}
    fill!(y_bar, zero(eltype(y_bar)))
    ptr = 0
    w_bar = Vector{Float64}()
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        resize!(w_bar, length(cell.data.points))
        fill!(w_bar, 0.0)
        for i = 1:length(cell.data.points)
            # H[cell.data.points[i]] += w[i]
            w_bar[i] += H_bar[cell.data.points[i]]
        end
        num_dof = size(Z[c],2)
        # w = wp[c] + Z[c]*y[ptr+1:ptr+num_dof]
        y_bar[ptr+1:ptr+num_dof] += Z[c]'*w_bar
        ptr += num_dof
   end
   return nothing
end

"""
    obj = quad_penalty(root, wp, Z, y, H_tol)

Compute the objective that seeks positive quadrature weights.  `root` is the
mesh, `wp` is a vector of particular solution to the cell-based quadrature
conditions, `y` is the null-space solution, `Z` is a vector of null-space
basis vectors on each cell, and `H_tol` is the target tolernace for each entry in the norm.
"""
function quad_penalty(root::Cell{Data, Dim, T, L}, wp, Z, y, H_tol
                      ) where {Data, Dim, T, L}
    # compute the diagonal norm based on the particular and null-space sol
    num_nodes = length(H_tol)
    H = zeros(eltype(y), num_nodes)
    diagonal_norm!(H, root, wp, Z, y)
    # add the penalties
    phi = 0.0
    for i = 1:num_nodes
        if real(H[i]) < H_tol[i]
            phi += 0.5*(H[i]/H_tol[i] - 1)^2
        end 
    end
    return phi
end

"""
    quad_penalty_grad!(g, root, wp, Z, y, H_tol)

Computes the derivative of `quad_penalty` with respect to `y`.  See
`quad_penalty` for an explanation of the remaining parameters.
"""
function quad_penalty_grad!(g::AbstractVector{T}, root::Cell{Data, Dim, T, L},
                       wp, Z, y, H_tol)  where {Data, Dim, T, L}
    num_nodes = length(H_tol)
    # need the diagonal norm for the reverse sweep 
    H = zeros(eltype(y), num_nodes)
    diagonal_norm!(H, root, wp, Z, y)
    # start the reverse sweep 
    fill!(g, zero(T))

    # return phi 
    phi_bar = 1.0 

    # add the penalties    
    H_bar = zero(H)
    for i = 1:num_nodes         
        if real(H[i]) < H_tol[i] 
            # phi += 0.5*(H[i]/H_tol[i] - 1)^2
            H_bar[i] += phi_bar*(H[i]/H_tol[i] - 1)/H_tol[i]
        end 
    end

    # diagonal_norm!(H, root, wp, Z, y)
    y_bar = g
    diagonal_norm_rev!(y_bar, H_bar, root, Z)
    return nothing
end

"""
    apply_approx_inverse!(p, g, root, wp, Z, y, H_tol, max_rank)

Finds a low-rank approximation to the inverse Hessian of the objective
`quad_penalty`, and applies this to `-g` to get a search direction `p`.  The
background mesh is `root`, `wp` is a vector of particular cell quadratures,
`Z` is the cell-quadrature null-space, and `y` is a vector of cell null-space
coefficients.  The rank of the approximation is controlled by `max_rank`.
"""
function apply_approx_inverse!(p::AbstractVector{T}, g::AbstractVector{T},
                               root::Cell{Data, Dim, T, L}, wp, Z, y, H_tol,
                               max_rank)  where {Data, Dim, T, L}
    num_nodes = length(H_tol)
    # need the diagonal norm for various steps
    H = zero(H_tol)
    diagonal_norm!(H, root, wp, Z, y)

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
    H_Jac = zeros(length(y), length(small_H))
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        num_dof = size(Z[c], 2)
        # first, check if any of the nodes in this cell's stencil are among the 
        # small_H indices 
        cell_set = intersect(small_H, cell.data.points)
        if length(cell_set) > 0
            # partial H/partial y
            for i = 1:length(cell_set)
                idx = findfirst(j -> j == cell_set[i], cell.data.points)
                col = findfirst(j -> j == cell_set[i], small_H)
                H_Jac[ptr+1:ptr+num_dof,col] += Z[c][idx,:]
            end
        end
        ptr += num_dof
    end

    # Compute the QR factorization of H_Jac; could do this in place to save 
    # memory
    A = H_Jac
    A *= diagm(diag_scal)
    F = qr(A)

    # find the search direction 
    Q = Matrix(F.Q)
    p[:] = -Q *( (F.R*F.R')\(Q'*g) )
    
    # for directions perpendicular to Q
    #p[:] -= 1e-4*(g - Q*(Q'*g))

    return nothing
end

function local_opt!(p::AbstractVector{T}, g::AbstractVector{T},
                    root::Cell{Data, Dim, T, L}, wp, Z, y, H_tol
                    )  where {Data, Dim, T, L}
    H = zero(H_tol)
    diagonal_norm!(H, root, wp, Z, y)
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        fac = zeros(length(cell.data.points))
        for (i,idx) in enumerate(cell.data.points)
            if H[idx] < H_tol[idx]
                fac[i] = 1/H_tol[idx]
            end
        end
        num_dof = size(Z[c],2)
        Hess = Z[c]'*((fac.*fac).*Z[c])
        p[ptr+1:ptr+num_dof] = -Hess \ g[ptr+1:ptr+num_dof]
        # A = fac.*Z[c]
        # F = qr(A')
        # Q = Matrix(F.Q)
        # p[ptr+1:ptr+num_dof] = -Q *( (F.R*F.R')\(Q'g[ptr+1:ptr+num_dof]) )
        ptr += num_dof
    end
end

"""
    H, w = opt_norm_null!(root, xc, degree, H_tol, max_rank)

Finds an approximate minimizer for the objective `quad_penalty` with respect to
the null-space coefficients.  The null-space is with respect to the cell
quadrature conditions, so there must be more nodes than quadrature conditions
for this to have a non-trivial solution.  The node coordinates are given by `xc`
and `root` is the background mesh.  Once an approximate solution is found, the
corresponding `H` norm is returned.  The parameter `max_rank` determines the
rank of the approximate Jacobian of the norm with respective to the null-space
coefficients, and this approximate Jacobian is used to form an approximate
inverse Hessian.
"""
function opt_norm_null!(root::Cell{Data, Dim, T, L}, xc, degree, H_tol, max_rank;
                       hist_file::Union{String,Nothing}=nothing, verbose::Bool=false
                       ) where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    max_iter = 1000
    max_line = 10
    H = zeros(num_nodes)
    # get the particular quadrature rules and null space vectors
    wp, Z, num_var = get_null_and_part(root, xc, degree)
    y = zeros(num_var)
    g = zero(y)
    p = zero(y) 
    if hist_file !== nothing 
       f = open(hist_file, "w")
       println(f,"# Norm optimization history file")
       println(f,"# degree = $degree")
       println(f,"# max_rank = $max_rank")
       println(f,"#")
       println(f,"# iter  obj  norm(g)  min(H)")
    end
    iter = 0
    for d = degree:degree #1:degree
        if verbose
            println(repeat("*",80))
            println("Starting optimization with degree = ",d)
        end
        #if d > 1
        #    H_tol[:] = 0.5*H[:]
        #end
        for n = 1:max_iter
            iter += 1
            obj = CloudSBP.quad_penalty(root, wp, Z, y, H_tol)
            CloudSBP.quad_penalty_grad!(g, root, wp, Z, y, H_tol)
            CloudSBP.diagonal_norm!(H, root, wp, Z, y)

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
            
            #fill!(p, 0.0)
            #println("norm(p) before = ",norm(p))
            CloudSBP.apply_approx_inverse!(p, g, root, wp, Z, y, H_tol, max_rank)
            #CloudSBP.local_opt!(p, g, root, wp, Z, y, H_tol)
            #println("norm(p) after = ",norm(p))
            #y[:] += p[:]
            #CloudSBP.diagonal_norm!(H, root, wp, Z, y)
            #break

            alpha = 1.0
            obj0 = obj
            for k = 1:max_line
                y[:] += alpha*p
                obj = CloudSBP.quad_penalty(root, wp, Z, y, H_tol)
                if verbose 
                    println("\t\tline-search iter ",k,": alpha = ",alpha,
                            ": obj0 = ",obj0,": obj = ",obj)
                end
                if obj < obj0
                    break
                end
                y[:] -= alpha*p
                alpha *= 0.1
            end
        end
    end # degree loop
    if hist_file !== nothing 
        close(f)
    end
    # compute the cell quadratures
    num_cell = num_leaves(root)
    w = Vector{Vector{T}}(undef, num_cell)
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        # compute the cell quadrature rule
        num_dof = size(Z[c],2)
        w[c] = wp[c] + Z[c]*y[ptr+1:ptr+num_dof]
        ptr += num_dof
    end
    return H, w
end

function positive_norm_cell!(Hc, wc, Zc, Hc_tol)
    @assert( length(Hc) == length(wc) == size(Zc,1) == length(Hc_tol),
            "Hc/wc/Zc/Hc_tol are inconsistent")
    if size(Zc,2) == 0
        # no freedom here to do anything
        return nothing
    end
    neg = count(i -> Hc[i] < Hc_tol[i] + eps(), axes(Hc,1))
    if neg == 0
        # none of the elements in Hc are negative, so nothing to do
        return nothing 
    end
    pos = length(Hc) - neg
    Zn = zeros(neg, size(Zc,2))
    Zp = zeros(pos, size(Zc,2))
    b = zeros(neg)
    c = zeros(pos)
    ptrn = 1
    ptrp = 1
    for i in axes(Hc,1)
        if Hc[i] < Hc_tol[i] + eps()
            scale = 1.0
            Zn[ptrn,:] = scale*(Zc[i,:]/Hc_tol[i])
            b[ptrn] = scale*(Hc[i]/Hc_tol[i] - 1)
            ptrn += 1
        else 
            Zp[ptrp,:] = Zc[i,:]/Hc_tol[i]
            c[ptrp] = Hc[i]/Hc_tol[i] - 1
            ptrp += 1
        end
    end
    if neg > size(Zc,2)
        # more neg nodes than DOF
        A = Zn'*Zn
        w = -A\(Zn'*b)
    else 
        # more DOF than neg nodes
        F = qr(Zn')
        Q = Matrix(F.Q)
        w = -Q *( (F.R*F.R')\(Q'*b) )
    end
    # now find the step length; this is conservative, because we stop when
    # we encouter the boundary
    alpha = 1.0
    for i in axes(Zp,1)
        ai = -c[i]/dot(Zp[i,:], w)
        if ai > 0
            alpha = min(alpha, ai)
        end
    end
    w .*= alpha 
    wc[:] += Zc*w
    Hc[:] += Zc*w
    return nothing
end

function calc_direction!(p, root::Cell{Data, Dim, T, L}, H, H_tol, Z
                         )  where {Data, Dim, T, L}
    fill!(p, 0.0)
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        Hc = view(H, cell.data.points)
        Hc_tol = view(H_tol, cell.data.points)
        num_dof = size(Z[c],2)
        for i in axes(Hc,1)
            if Hc[i] < Hc_tol[i]
                p[ptr+1:ptr+num_dof] += Z[c][i,:]/Hc_tol[i]
            end
        end
        ptr += num_dof
    end
    return nothing
end

function calc_step_length(p, root::Cell{Data, Dim, T, L}, H, H_tol, Z
                          )  where {Data, Dim, T, L}
    Zdotg = zeros(length(H))
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        Hc = view(H, cell.data.points)
        Hc_tol = view(H_tol, cell.data.points)
        num_dof = size(Z[c],2)
        for i in axes(Hc,1)
            Zdotg[cell.data.points[i]] += dot(Z[c][i,:], p[ptr+1:ptr+num_dof])
        end
        ptr += num_dof
    end
    alpha = 1.0
    for i in axes(H,1)
        ai = (H_tol[i] - H[i])/Zdotg[i]
        if H[i] >= H_tol[i] && ai > 0.0
            alpha = min(ai, alpha)
        end
    end
    return alpha
end

function positive_norm!(root::Cell{Data, Dim, T, L}, xc, degree, H_tol;
                        verbose::Bool=false) where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    H = zeros(num_nodes)
    # get the particular quadrature rules and null space vectors
    wp, Z, num_var = get_null_and_part(root, xc, degree)
    y = zeros(num_var)
    p = zero(y)

    max_iter = 100
    for k = 1:max_iter
        CloudSBP.diagonal_norm!(H, root, wp, Z, y)
        if verbose
            min_idx = argmin(H)
            neg = count(i -> H[i] < H_tol[i], axes(H,1))
            println("iter $k: minimum(H) = ",H[min_idx],": num_neg = ",neg)
        end

        # get the step direction 
        calc_direction!(p, root, H, H_tol, Z)

        # get the step length
        alpha = calc_step_length(p, root, H, H_tol, Z)
        println("\talpha = $alpha")

        # update the particular solution
        ptr = 0
        for (c, cell) in enumerate(allleaves(root))
            if is_immersed(cell)
                continue
            end
            num_dof = size(Z[c],2)
            wp[c] += alpha*Z[c]*p[ptr+1:ptr+num_dof]
            ptr += num_dof
        end
    end

    return H, wp
end

"""
    y, s, lam = view_vars(x, num_nodes, num_vars)

Extracts views of the primal (`y`), slacks (`s`), and multipliers (`lam`) from
the given compound vector `x`.  `num_nodes` and `num_vars` are the number of
nodes and the number of degrees of freedom, respectively.
"""
function view_vars(x, num_nodes, num_vars)
    @assert( length(x) == num_vars + 2*num_nodes, "length(x) is inconsistent" )
    y = view(x, 1:num_vars)
    s = view(x, num_vars+1:num_vars+num_nodes)
    lam = view(x, num_vars+num_nodes+1:num_vars+2*num_nodes)
    return y, s, lam
end

"""
    prim, comp, feas = first_order_opt!(g, x, root, wp, Z, H_tol)

Computes the first-order optimality based on the variables in `x` and stores
the KKT conditions in `g`.  `root` stores the mesh and `wp` and `Z` define the
particular norm and its null-space over the cells.  `H_tol` is the desired
tolerance for the norms.  The function returns the norm of the components of
`g`, namely primal optimality, complementarity, and feasibility.
"""
function first_order_opt!(g, x, root, wp, Z, H_tol)
    num_nodes = length(H_tol)
    num_vars = sum(Zc -> size(Zc,2), Z)
    y, s, lam = view_vars(x, num_nodes, num_vars)
    gy, gs, glam = view_vars(g, num_nodes, num_vars)
    
    gs[:] = s.*lam
    glam[:] = s .+ 1.0
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        num_dof = size(Z[c],2)
        slice = ptr+1:ptr+num_dof
        lam_c = lam[cell.data.points]
        H_tol_c = H_tol[cell.data.points]
        gy[slice] = y[slice] - Z[c]'*(lam_c./H_tol_c)
        glam[cell.data.points] -= (wp[c] + Z[c]*y[slice])./H_tol_c
        ptr += num_dof
    end
    return norm(gy), norm(gs), norm(glam)
end

"""
    kkt_vector_product!(v, u, x, root, Z, H_tol)

Applies the KKT matrix, evaluated at `x`, to the compound vector `u` and sets
the product to the compound vector `v`.  `root` stores the mesh and `wp` and `Z`
define the particular norm and its null-space over the cells.  `H_tol` is the
desired tolerance for the norms.
"""
function kkt_vector_product!(v, u, x, root, Z, H_tol)
    num_nodes = length(H_tol)
    num_vars = sum(Zc -> size(Zc,2), Z)
    y, s, lam = view_vars(x, num_nodes, num_vars)
    uy, us, ulam = view_vars(u, num_nodes, num_vars)
    vy, vs, vlam = view_vars(v, num_nodes, num_vars)
    # complementarity
    vs[:] = lam.*us + s.*ulam 
    # loop for primal and constraint portions
    vlam[:] = us
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        num_dof = size(Z[c],2)
        slice = ptr+1:ptr+num_dof
        ulam_c = ulam[cell.data.points]
        H_tol_c = H_tol[cell.data.points]
        # primal optimality
        vy[slice] = uy[slice] - Z[c]'*(ulam_c./H_tol_c)
        # add to constraint
        vlam[cell.data.points] -= (Z[c]*uy[slice])./H_tol_c
        ptr += num_dof
    end
    return nothing
end

"""
    precond_interior_point!(v, u, x, root, Z, H_tol, diagZZt)

Applies a preconditioner (based on the solution in `x`) to the compound vector
`u` and sets the result to the compound vector `v`.  `root` stores the mesh and
`wp` and `Z` define the particular norm and its null-space over the cells.
`H_tol` is the desired tolerance for the norms.  Finally, `diagZZt` is the
diagonal part of the matrix `Z*Z'`.
"""
function precond_interior_point!(v, u, x, root, Z, H_tol, diagZZt)
    num_nodes = length(H_tol)
    num_vars = sum(Zc -> size(Zc,2), Z)
    y, s, lam = view_vars(x, num_nodes, num_vars)
    uy, us, ulam = view_vars(u, num_nodes, num_vars)
    vy, vs, vlam = view_vars(v, num_nodes, num_vars)
    # for the RHS for the multiplier system
    lam_rhs = us - lam.*ulam
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        num_dof = size(Z[c],2)
        slice = ptr+1:ptr+num_dof
        lam_c = lam[cell.data.points]
        H_tol_c = H_tol[cell.data.points]
        lam_rhs[cell.data.points] -= (lam_c./H_tol_c).*(Z[c]*uy[slice])
        ptr += num_dof
    end
    vlam = lam_rhs./(s + lam.*diagZZt)
    # solve for the primal and slack
    vs[:] = ulam
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        num_dof = size(Z[c],2)
        slice = ptr+1:ptr+num_dof
        vlam_c = vlam[cell.data.points]
        H_tol_c = H_tol[cell.data.points]
        # primal optimality
        vy[slice] = uy[slice] + Z[c]'*(vlam_c./H_tol_c)
        # slack update
        vs[cell.data.points] += (Z[c]*vy[slice])./H_tol_c
        ptr += num_dof
    end
    return nothing
end

"""
    diagZZt = compute_diag_ZZt(root, Z, H_tol)

Returns the diagonal of `Z * Z^T`, with the rows of `Z` scaled by `H_tol`.
NOTE: while ZZ^T is SPD, diag(ZZ^T) may not be. 
"""
function compute_diag_ZZt(root, Z, H_tol)
    num_nodes = length(H_tol)
    diagZZt = zeros(num_nodes)
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        diagZZt_c = diag(Z[c]*Z[c]')./(H_tol[cell.data.points].^2)
        diagZZt[cell.data.points] += diagZZt_c
    end
    for d in diagZZt
        @assert( d > eps() )
    end
    return diagZZt
end

"""
    initial_guess!(x, root, wp, Z, H_tol, diagZZt)

Sets the compound vector `x` to a reasonable initial guess for the primal,
slacks, and multipliers.  `root` defines the mesh, and `wp` and `Z` are
(cell-based) particular and null-space solutions to the norm equations.
`H_tol` is the tolerance for the norm at each node, and `diagZZt` is the
diagonal part of `Z*Z^T`
"""
function initial_guess!(x, root, wp, Z, H_tol, diagZZt)
    num_nodes = length(H_tol)
    num_vars = sum(Zc -> size(Zc,2), Z)
    fill!(x, 0.0)
    y, s, lam = view_vars(x, num_nodes, num_vars)
    # use s to store the right-hand side of the constraint
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        s[cell.data.points] += wp[c]./H_tol[cell.data.points]
    end
    s[:] .-= 1.0 
    # solve for slacks and multipliers
    for i in axes(s,1)
        b = s[i]
        s[i] = b/(1.0 + diagZZt[i])
        lam[i] =(s[i] - b)/diagZZt[i]
    end
    # solve for the primal solution
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        num_dof = size(Z[c],2)
        slice = ptr+1:ptr+num_dof
        y[slice] = -Z[c]'*s[cell.data.points]
        ptr += num_dof
    end
    # enforce positivity of s and lambda
    delta_s = max(-(1.5)*minimum(s), 0.0)
    s[:] .+= delta_s
    delta_lam = max(-(1.5)*minimum(lam), 0.0)
    lam[:] .+= delta_lam
    # move s and lambda away from zero and balance
    s_dot_lam = dot(s, lam)
    delta_s = 0.5*s_dot_lam/sum(lam)
    delta_lam = 0.5*s_dot_lam/sum(s)
    s[:] .+= delta_s
    lam[:] .+= delta_lam
    return nothing
end

function solve_norm(root, xc, degree, H_tol; verbose::Bool=false,
                    max_iter::Int=100, rel_tol::Float64=1e-8, abs_tol::Float64=1e-10,
                    max_krylov::Int=50)

    # get the particular quadrature rules and null space vectors
    wp, Z, num_vars = get_null_and_part(root, xc, degree)
    num_nodes = length(H_tol)
    n = num_vars + 2*num_nodes

    # get memory for some vectors 
    x = zeros(num_vars + 2*num_nodes)
    y, s, lam = view_vars(x, num_nodes, num_vars)
    dx = zero(x)
    dy, ds, dlam = view_vars(dx, num_nodes, num_vars)
    g = zero(x)
    gy, gs, glam = view_vars(g, num_nodes, num_vars)

    # get the diagonal preconditioner and a positive initial guess 
    diagZZt = compute_diag_ZZt(root, Z, H_tol)
    initial_guess!(x, root, wp, Z, H_tol, diagZZt)

    if verbose
        println("# iter.   opt.   comp.   feas.")
    end
    prim0, comp0, feas0 = 1.0, 1.0, 1.0
    for k in 1:max_iter

        # check for convergence 
        prim, comp, feas = first_order_opt!(g, x, root, wp, Z, H_tol)
        if verbose 
            println("$k   $prim   $comp   $feas")
        end
        if k == 0
            prim0, comp0, feas0 = prim, comp, feas
        else 
            # check for relative convergence 
            if prim < prim0*rel_tol && comp < comp0*rel_tol && feas < feas0*rel_tol 
                break
            end
        end
        # check for absolute convergence 
        if prim < abs_tol && comp < abs_tol && feas < abs_tol 
            break
        end
        if k == max_iter
            error("Interior point algorithm failed to converge")
        end

        # find the affine (Newton) step
        A = LinearOperator(Float64, n, n, false, false, (v, u) -> 
                           kkt_vector_product!(v, u, x, root, Z, H_tol))
        P = LinearOperator(Float64, n, n, false, false, (v, u) ->
                           precond_interior_point!(v, u, x, root, Z, H_tol, diagZZt))
        g .*= -1.0
        dx[:], stats = gmres(A, g, memory=max_krylov, atol=1e-12, rtol=1.0e-10, 
                          verbose=Int(verbose), N=P, reorthogonalization=true)

        # find mu affine and the centering parameter sigma
        alpha_s = 1.0
        alpha_lam = 1.0
        for i in axes(s,1)
            ds[i] < 0.0 ? alpha_s = min(alpha_s, -s[i]/ds[i]) : nothing
            dlam[i] < 0.0 ? alpha_lam = min(alpha_lam, -lam[i]/dlam[i]) : nothing
        end 
        mu = dot(s, lam)/num_nodes # same as comp^2/num_nodes
        mu_aff = 0.0
        for i in axes(s,1)
            mu_aff += (s[i] + alpha_s*ds[i])*(lam[i] + alpha_lam*dlam[i])
        end
        mu_aff /= num_nodes 
        sigma = (mu_aff/mu)^3 
        println("mu = ",mu)
        println("mu_aff = ",mu_aff)
        println("sigma = ",sigma)

        # update the RHS of the linear system and solve again 
        for i in axes(gs,1)
            gs[i] -= ds[i]*dlam[i] + sigma*mu
        end
        dx[:], stats = gmres(A, g, memory=max_krylov, atol=1e-12, rtol=1.0e-10, 
                          verbose=Int(verbose), N=P, reorthogonalization=true)
        
        # find the step lengths and update solution
        eta = 0.9 #exp(-comp) + (1 - exp(-comp))*0.9
        alpha_s = 1.0
        alpha_lam = 1.0
        for i in axes(s,1)
            ds[i] < 0.0 ? alpha_s = min(alpha_s, -eta*s[i]/ds[i]) : nothing
            dlam[i] < 0.0 ? alpha_lam = min(alpha_lam, -eta*lam[i]/dlam[i]) : nothing
        end
        y[:] += alpha_s*dy[:]
        s[:] += alpha_s*ds[:]
        lam[:] += alpha_lam*dlam[:]
    end
    # Compute the norm and update the particular solution
    CloudSBP.diagonal_norm!(H, root, wp, Z, y)
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        num_dof = size(Z[c],2)
        wp[c] += Z[c]*y[ptr+1:ptr+num_dof]
        ptr += num_dof
    end
    return H, wp, Z
end
