# Tests related to node optimization for the norm

@testset "test cell_quadrature_rev!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = num_basis
    cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))
    num_cells = 1
    cell.data.xref = 0.5*ones(Dim)
    cell.data.dx = ones(Dim)
    m = CloudSBP.calc_moments!(cell, degree)
    xc = randn(Dim, num_nodes) .+ 0.5

    # compute the derivative of the (weighted) quad weights w.r.t. xc 
    w_bar = randn(num_nodes)
    xc_bar = zero(xc)
    CloudSBP.cell_quadrature_rev!(xc_bar, degree, xc, m, cell.data.xref,
                                cell.data.dx, w_bar, Val(Dim))
    p = randn(size(xc_bar))
    dot_prod = dot(vec(xc_bar), vec(p))

    # now use complex step to approximate the same derivatives
    ceps = 1e-60
    xc_cmplx = complex.(xc, ceps*p)
    w_cmplx = CloudSBP.cell_quadrature(degree, xc_cmplx, m, cell.data.xref,
                                     cell.data.dx, Val(Dim))
    dot_prod_cmplx = dot(w_bar, imag.(w_cmplx)/ceps)

    @test isapprox(dot_prod, dot_prod_cmplx)
end

@testset "test cell_quadrature_rev! (old): dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = 2*num_basis
    x1d, w1d = CloudSBP.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}())) #Face{2,Float64}}()))
    CloudSBP.quadrature!(xq, wq, cell.boundary, x1d, w1d)
    xc = randn(Dim, num_nodes) .+ 0.5

    # compute the derivative of the (weighted) quad weights w.r.t. xc 
    w_bar = randn(num_nodes)
    xc_bar = zero(xc)
    CloudSBP.cell_quadrature_rev!(xc_bar, degree, xc, xq, wq, w_bar, Val(Dim))
    p = randn(size(xc_bar))
    dot_prod = dot(vec(xc_bar), vec(p))

    # now use complex step to approximate the same derivatives
    ceps = 1e-60
    xc_cmplx = complex.(xc, ceps*p)
    w_cmplx = CloudSBP.cell_quadrature(degree, xc_cmplx, xq, wq, Val(Dim))
    dot_prod_cmplx = dot(w_bar, imag.(w_cmplx)/ceps)

    @test isapprox(dot_prod, dot_prod_cmplx)
end

@testset "test diagonal_norm_rev!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # DGD dof locations
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = 5*num_basis
    points = rand(Dim, num_nodes)
    
    # refine mesh, build sentencil, and evaluate norm
    CloudSBP.refine_on_points!(root, points)
    CloudSBP.build_nn_stencils!(root, points, degree)
    CloudSBP.set_xref_and_dx!(root, points)
    m = CloudSBP.calc_moments!(root, degree)

    # get the derivative of dot(H_bar, H) in direction p
    H_bar = randn(num_nodes)
    points_bar = zero(points)
    CloudSBP.diagonal_norm_rev!(points_bar, H_bar, root, points, degree)
    p = randn(Dim, num_nodes)
    dot_prod = dot(vec(points_bar), vec(p))

    # get the derivative using complex step
    ceps = 1e-60
    points_cmplx = complex.(points, ceps*p)
    H_cmplx = zeros(ComplexF64, num_nodes)
    CloudSBP.diagonal_norm!(H_cmplx, root, points_cmplx, degree) 
    dot_prod_cmplx = dot(H_bar, imag.(H_cmplx)/ceps)

    @test isapprox(dot_prod, dot_prod_cmplx)
end

@testset "test penalty_grad!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # DGD dof locations
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = 5*num_basis
    points = rand(Dim, num_nodes)
    points_init = points + 0.05*rand(Dim, num_nodes)

    # refine mesh and build stencil
    CloudSBP.refine_on_points!(root, points)
    CloudSBP.build_nn_stencils!(root, points, degree)
    CloudSBP.set_xref_and_dx!(root, points)
    m = CloudSBP.calc_moments!(root, degree)

    # compute the penalty gradient 
    mu = 1.0
    H_tol = 1e-3.*ones(num_nodes)
    g = zeros(Dim*num_nodes)
    dist_ref = 1.0 .+ 0.1*rand(num_nodes)
    CloudSBP.penalty_grad!(g, root, points, points_init, dist_ref, H_tol, mu,
                         degree)

    # compare against a complex-step based directional derivative 
    p = randn(length(g))
    gdotp = dot(g, p)

    ceps = 1e-60
    points_cmplx = complex.(points, ceps.*reshape(p, size(points)))
    penalty = CloudSBP.penalty(root, points_cmplx, points_init, dist_ref, H_tol,
                             mu, degree)
    gdotp_cmplx = imag(penalty)/ceps 

    @test isapprox(gdotp, gdotp_cmplx)
end

@testset "test obj_norm_grad!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # DGD dof locations
    num_basis = binomial(Dim + degree, Dim)

    num_nodes = 5*num_basis
    points = rand(Dim, num_nodes)

    # refine mesh, build sentencil, and evaluate particular quad and nullspace
    CloudSBP.refine_on_points!(root, points)
    CloudSBP.build_nn_stencils!(root, points, degree)
    CloudSBP.set_xref_and_dx!(root, points)
    m = CloudSBP.calc_moments!(root, degree)
    wp, Z, num_var = CloudSBP.get_null_and_part(root, points, degree)

    # compute objective gradient for random y 
    rho = 1.0
    y = 0.1*randn(num_var)
    g = zero(y)
    CloudSBP.obj_norm_grad!(g, root, wp, Z, y, rho, num_nodes)

    # compare against a complex-step based directional derivative 
    p = randn(num_var)
    gdotp = dot(g, p)

    ceps = 1e-60
    num_cell = CloudSBP.num_leaves(root)
    wp_cmplx = Vector{Vector{ComplexF64}}(undef, num_cell)
    Z_cmplx = Vector{Matrix{ComplexF64}}(undef, num_cell)
    for i = 1:num_cell 
        wp_cmplx[i] = complex.(wp[i])
        Z_cmplx[i] = complex.(Z[i])
    end
    y_cmplx = complex.(y, ceps.*p)
    rho_cmplx = complex(rho)
    obj = CloudSBP.obj_norm(root, wp_cmplx, Z_cmplx, y_cmplx, rho_cmplx,
                          num_nodes)
    gdotp_cmplx = imag(obj)/ceps 

    @test isapprox(gdotp, gdotp_cmplx)

end