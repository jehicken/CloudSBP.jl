
@testset "test diagonal_norm!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    
    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
    SVector(ntuple(i -> 1.0, Dim)),
    CellData(Vector{Int}(), Vector{Int}()))
    
    # DGD dof locations
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = 5*num_basis
    points = rand(Dim, num_nodes)
    
    # refine mesh, build stencil, and evaluate norm
    CutDGD.refine_on_points!(root, points)
    CutDGD.build_nn_stencils!(root, points, degree)
    CutDGD.set_xref_and_dx!(root, points)
    m = CutDGD.calc_moments!(root, degree)
    wp, Z, num_var = CutDGD.get_null_and_part(root, points, degree)
    H = zeros(num_nodes)
    y = 0.1*randn(num_var)
    CutDGD.diagonal_norm!(H, root, wp, Z, y)
    
    # get quadrature to compute reference integrals
    x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    CutDGD.quadrature!(xq, wq, root.boundary, x1d, w1d)
    
    # compare H-based quadrature with LG-based quadrature
    V = zeros(num_nodes, num_basis)
    CutDGD.monomial_basis!(V, degree, points, Val(Dim))
    Vq = zeros(num_quad, num_basis)
    CutDGD.monomial_basis!(Vq, degree, xq, Val(Dim))
    for k = 1:num_basis
        integral_ref = dot(Vq[:,k], wq)
        integral = dot(V[:,k], H)
        @test isapprox(integral, integral_ref)
    end
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
    CutDGD.refine_on_points!(root, points)
    CutDGD.build_nn_stencils!(root, points, degree)
    CutDGD.set_xref_and_dx!(root, points)
    m = CutDGD.calc_moments!(root, degree)
    wp, Z, num_var = CutDGD.get_null_and_part(root, points, degree)

    # get the derivative of dot(H_bar, H) in direction p
    H_bar = randn(num_nodes)
    y_bar = zeros(num_var)
    CutDGD.diagonal_norm_rev!(y_bar, H_bar, root, Z)
    p = randn(num_var)
    dot_prod = dot(y_bar, p)

    # get the derivative using complex step
    ceps = 1e-60
    y = randn(num_var) # H is linear in y, so values are irrelevant here
    y_cmplx = complex.(y, ceps*p)
    H_cmplx = zeros(ComplexF64, num_nodes)
    CutDGD.diagonal_norm!(H_cmplx, root, wp, Z, y_cmplx)
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
    CutDGD.refine_on_points!(root, points)
    CutDGD.build_nn_stencils!(root, points, degree)
    CutDGD.set_xref_and_dx!(root, points)
    m = CutDGD.calc_moments!(root, degree)
    wp, Z, num_var = CutDGD.get_null_and_part(root, points, degree)

    # compute the penalty gradient 
    H_tol = 1e-3.*ones(num_nodes)
    g = zeros(num_var)
    y = randn(num_var)
    CutDGD.quad_penalty_grad!(g, root, wp, Z, y, H_tol)

    # compare against a complex-step based directional derivative 
    p = randn(length(g))
    gdotp = dot(g, p)

    ceps = 1e-60
    y_cmplx = complex.(y, ceps.*p)
    penalty = CutDGD.quad_penalty(root, wp, Z, y_cmplx, H_tol)
    gdotp_cmplx = imag(penalty)/ceps 

    @test isapprox(gdotp, gdotp_cmplx)
end