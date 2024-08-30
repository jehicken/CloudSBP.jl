# Not used presently

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
    wp, Z, num_var = CloudSBP.get_null_and_part(root, points, degree)

    # get the derivative of dot(H_bar, H) in direction p
    H_bar = randn(num_nodes)
    y_bar = zeros(num_var)
    CloudSBP.diagonal_norm_rev!(y_bar, H_bar, root, Z)
    p = randn(num_var)
    dot_prod = dot(y_bar, p)

    # get the derivative using complex step
    ceps = 1e-60
    y = randn(num_var) # H is linear in y, so values are irrelevant here
    y_cmplx = complex.(y, ceps*p)
    H_cmplx = zeros(ComplexF64, num_nodes)
    CloudSBP.diagonal_norm!(H_cmplx, root, wp, Z, y_cmplx)
    dot_prod_cmplx = dot(H_bar, imag.(H_cmplx)/ceps)

    @test isapprox(dot_prod, dot_prod_cmplx)
end

@testset "test quad_penalty_grad!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # DGD dof locations
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = 5*num_basis
    points = rand(Dim, num_nodes)

    # refine mesh and build stencil
    CloudSBP.refine_on_points!(root, points)
    CloudSBP.build_nn_stencils!(root, points, degree)
    CloudSBP.set_xref_and_dx!(root, points)
    m = CloudSBP.calc_moments!(root, degree)
    wp, Z, num_var = CloudSBP.get_null_and_part(root, points, degree)

    # compute the penalty gradient 
    H_tol = 1e-3.*ones(num_nodes)
    g = zeros(num_var)
    y = randn(num_var)
    CloudSBP.quad_penalty_grad!(g, root, wp, Z, y, H_tol)

    # compare against a complex-step based directional derivative 
    p = randn(length(g))
    gdotp = dot(g, p)

    ceps = 1e-60
    y_cmplx = complex.(y, ceps.*p)
    penalty = CloudSBP.quad_penalty(root, wp, Z, y_cmplx, H_tol)
    gdotp_cmplx = imag(penalty)/ceps 

    @test isapprox(gdotp, gdotp_cmplx)
end

@testset "test kkt_vector_product!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # DGD dof locations
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = 5*num_basis
    points = rand(Dim, num_nodes)

    # refine mesh and build stencil
    CloudSBP.refine_on_points!(root, points)
    CloudSBP.build_nn_stencils!(root, points, degree)
    CloudSBP.set_xref_and_dx!(root, points)
    m = CloudSBP.calc_moments!(root, degree)
    wp, Z, num_vars = CloudSBP.get_null_and_part(root, points, degree)

    # evaluate product for random x and u
    x = randn(num_vars + 2*num_nodes)
    u = randn(num_vars + 2*num_nodes)
    v = zero(u)
    H_tol = 1e-3.*ones(num_nodes)
    CloudSBP.kkt_vector_product!(v, u, x, root, Z, H_tol)

    # now compute product using complex step 
    ceps = 1e-60
    g = zeros(ComplexF64, length(x))
    x_cmplx = complex.(x, ceps.*u)
    prim, comp, feas = CloudSBP.first_order_opt!(g, x_cmplx, root, wp, Z, H_tol)

    v_cmplx = imag.(g)/ceps
    @test isapprox(v, v_cmplx)
end