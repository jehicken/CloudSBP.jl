# Routines related to constructing a diagonal norm

"""
    m = calc_moments!(root, levset, degree, fit_degree)

Returns the first total `degree` integral moments for all cells in the tree
defined by `root`.  The tree must have been preprocessed to identify poentially
cut and immersed cells using the `levset` level-set function.  In addition, the 
`cell.data.xref` and `cell.data.dx` fields must contain the reference origin
for each cell.

WARNING: The signature of the function `levset` must of the form
levset(Vector{Float64})::Float64, because this assumption is used when it is
wrapped using `csafe_function`.
"""
function calc_moments!(root::Cell{Data, Dim, T, L}, levset, degree, fit_degree
                       ) where {Data, Dim, T, L}
    @assert( degree >= 0, "degree must be non-negative" )
    @assert( 0 <= fit_degree <= degree, "invalid fit_degree value" )
    num_cell = num_leaves(root)
    num_basis = binomial(Dim + degree, Dim)
    moments = zeros(num_basis, num_cell)

    # get arrays/data used for tensor-product quadrature 
    num_quad1d = ceil(Int, (degree+1)/2)
    x1d, w1d = lg_nodes(num_quad1d) # could also use lgl_nodes
    num_quad = length(w1d)^Dim
    wq = zeros(num_quad)
    xq = zeros(Dim, num_quad)
    Vq = zeros(num_quad, num_basis)
    workq = zeros((Dim+1)*num_quad)

    for (c, cell) in enumerate(allleaves(root))
        xref = cell.data.xref 
        dx = cell.data.dx 
        if is_immersed(cell)
            # do not integrate cells that have been confirmed immersed
            continue
        elseif is_cut(cell)
            # this cell *may* be cut; use Saye's algorithm
            wq_cut, xq_cut = cut_cell_quad(cell.boundary, levset, num_quad1d, 
                                           fit_degree=fit_degree)
            # consider resizing 1D arrays here, if need larger
            for I in CartesianIndices(xq_cut)
                xq_cut[I] = (xq_cut[I] - xref[I[1]])/dx[I[1]] - 0.5
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
                xq[I] = (xq[I] - xref[I[1]])/dx[I[1]] - 0.5
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
    m = calc_moments!(root, degree)

This variant is useful when you want moments for all cells in the tree `root`, 
i.e. no level-set function.
"""
function calc_moments!(root::Cell{Data, Dim, T, L}, degree
                       ) where {Data, Dim, T, L}
    levset(x) = 1.0
    return calc_moments!(root, levset, degree, 0)
end

"""
    w = cell_quadrature(degree, xc, xq, wq, Val(Dim))

Given a quadrature rule `(xq,wq)` that is exact for polynomials of `degree` 
degree, computes a new quadrature for the same domain based on the nodes `xc`. 
This version computes the moments from scratch.
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

"""
    w = cell_quadrature(degree, xc, moments, xref, dx, Val(Dim))

Given a set of total `degree` polynomial `moments`, computes a quadrature that
is exact for those moments based on the nodes `xc`.  The arrays `xref` and `dx`
are used to shift and scale, respectively, the nodes in `xc` to improve 
conditioning of the Vandermonde matrix.  The same scaling and shifts must have
been applied when computing the integral `moments`.
"""
function cell_quadrature(degree, xc, moments, xref, dx, ::Val{Dim}) where {Dim}
    @assert( size(xc,1) == length(dx) == length(xref) == Dim,
             "xc/dx/xref/Dim are inconsistent")
    num_basis = binomial(Dim + degree, Dim)
    @assert( length(moments) >= num_basis, "moments has inconsistent size")
    num_nodes = size(xc, 2)
    if num_nodes < num_basis
        println("num_nodes = ",num_nodes)
        println("num_basis = ",num_basis)
    end
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
    solve_min_norm!(w, V, moments[1:num_basis])
    return w
end

"""
    diagonal_norm!(H, root, points, degree)

Constructs a diagonal norm (i.e. quadrature) for the nodes `points` and using
the background mesh defined by `root`.

**PRE**: The cells in `root` must have their `data.moments`, `data.xref`, and
`data.dx` fields set.
"""
function diagonal_norm!(H, root::Cell{Data, Dim, T, L}, points, degree
                       ) where {Data, Dim, T, L}
    @assert( size(points,1) == Dim, "points inconsistent with Dim" )  
    @assert( size(points,2) == length(H), "points inconsistent with H")
    fill!(H, zero(eltype(H)))
    for cell in allleaves(root)
        if is_immersed(cell)
            continue
        end
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

"""
    diagonal_norm!(H, root)

Assembles the diagonal norm `H` using the particular cell-based norm stored in 
the cells' `data.wts` field.  `root` stores the mesh.
"""
function diagonal_norm!(H, root)
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        # add the local contribution
        for i = 1:length(cell.data.points)            
            H[cell.data.points[i]] += cell.data.wts[i]
        end
   end
   return nothing
end

"""
    w = cell_null_and_part(degree, xc, moments, xref, dx, Val(Dim))

Given a set of total `degree` polynomial `moments`, computes a quadrature that
is exact for those moments based on the nodes `xc` as well as the null space of
the quadrature conditions  The arrays `xref` and `dx` are used to shift and
scale, respectively, the nodes in `xc` to improve conditioning of the
Vandermonde matrix.  The same scaling and shifts must have been applied when
computing the integral `moments`.
"""
function cell_null_and_part(degree, xc, moments, xref, dx, ::Val{Dim}) where {Dim}
    println("size(xc,1) = ",size(xc,1))
    println("length(dx) = ",length(dx))
    println("length(xref) = ",length(xref))
    println("Dim = ",Dim)
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
    solve_min_norm!(w, V, moments[1:num_basis])
    Z = nullspace(V')
    return w, Z
end

"""
    wp, Z, num_vars = get_null_and_part(root, xc, degree)

Returns a vector of nullspaces, `Z`, and particular solutions, `wp`, for each 
cell's quadrature problem.  Also returns the total number of degrees of 
freedom.  Note that `wp` and `Z` are `Vector`s of `Vector` and `Matrix`, 
respectively.

**PRE**: The cells in `root` must have their `data.moments`, `data.xref`, and
`data.dx` fields set.
"""
function get_null_and_part(root::Cell{Data, Dim, T, L}, xc, degree
                           ) where {Data, Dim, T, L}
    @assert( size(xc,1) == Dim, "xc coordinates inconsistent with Dim" )
    num_cell = num_leaves(root)
    Z = [zeros(T, (0,0)) for i in 1:num_cell]
    wp = [zeros(T, (0)) for i in 1:num_cell]
    num_vars = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        # get the nodes in this cell's stencil
        nodes = view(xc, :, cell.data.points)
        # get cell quadrature and add to global norm
        wp[c], Z[c] = cell_null_and_part(degree, nodes, cell.data.moments,
                                         cell.data.xref, cell.data.dx,
                                         Val(Dim))
        num_vars += size(Z[c], 2)
    end
    return wp, Z, num_vars
end

"""
    A, b = compute_sparse_constraint(root, wp, Z, H_tol)

Returns the sparse matrix `A` and vector `b` that make up the linear inequality 
for the diagonal mass matrix.  `root` is used to loop over the cells and 
assemble the null-space matrices stored in `Z` and the minimum-norm quadrature 
stored in `wp`.  `H_tol` is an array of tolerances for the diagonal entries.
"""
function compute_sparse_constraint(root, wp, Z, H_tol)
    num_nodes = length(H_tol)
    rows = Int[]
    cols = Int[]
    Avals = Float64[]
    b = zeros(num_nodes)
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        for i in axes(Z[c],1)
            for j in axes(Z[c],2)
                append!(rows, cell.data.points[i])
                append!(cols, ptr+j)
                append!(Avals, Z[c][i,j])
            end
            b[cell.data.points[i]] -= wp[c][i]
        end
        ptr += size(Z[c],2)
    end
    b[:] += H_tol
    return sparse(rows, cols, Avals), b
end

"""
    H, success = solve_norm!(root, xc, degree, H_tol [, verbose=false])

Attempts to solve the norm-inequality problem using the interior-point method 
implemented in the Tulip.jl package.  `root` defines the background mesh, `xc` 
are the nodes, degree is the target degree of the norm, and `H_tol` holds the 
lower bound on the norm entries.  If `verbose` is `true`, the `"OutputLevel"` 
attribute of Tulip is set to 1 (so more output is provided).  The method 
returns the diagonal norm, `H`, and a `Bool`, `success`, that indicates whether 
Tulip was successful (`success=true`) or not.  If the problem was not 
successfully solved, `H` is returned with the minimum-norm quadrature weights 
(which likely have some negative weights).

**PRE**: The cells in `root` must have their `data.moments`, `data.xref`, and
`data.dx` fields set.
"""
function solve_norm!(root, xc, degree, H_tol; verbose::Bool=false)
    @assert( size(xc,2) == length(H_tol), "xc and H_tol are inconsistent")
    H = zero(H_tol)
    # get the particular quadrature rules and null space vectors
    wp, Z, num_vars = get_null_and_part(root, xc, degree)
    num_nodes = length(H_tol)
    n = num_vars + 2*num_nodes
    y = zeros(num_vars)

    # get the sparse constraint, A*y >= b
    A, b = compute_sparse_constraint(root, wp, Z, H_tol)

    # set up the optimization problem
    model = Model(Tulip.Optimizer)
    set_attribute(model, "IPM_IterationsLimit", 20)
    if verbose
        set_attribute(model, "OutputLevel", 1)
    end
    @variable(model, y[1:num_vars])
    @objective(model, Min, 0.0)
    @constraint(model, A*y >= b)
    optimize!(model)
    #solution_summary(model)
    y = value.(y)
    success = true
    if !is_solved_and_feasible(model)
        y .= 0.0
        success = false
    end
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        if is_immersed(cell)
            continue
        end
        num_dof = size(Z[c],2)
        cell.data.wts = wp[c] + Z[c]*y[ptr+1:ptr+num_dof]
        ptr += num_dof
    end
    CloudSBP.diagonal_norm!(H, root)
    return H, success
end