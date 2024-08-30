# Test utility routines

@testset "test solve_min_norm_rev!" begin
    num_eq = 10
    num_dof = 50
    V = randn(num_dof,num_eq)
    w = zeros(num_dof)
    b = randn(num_eq)
    
    eps_fd = 1e-6
    CloudSBP.solve_min_norm!(w, V, b)
    V_pert = randn(size(V))
    w_pert = zero(w)
    CloudSBP.solve_min_norm!(w_pert, V + eps_fd*V_pert, b)
    w_bar = randn(size(w))
    dot_fd = dot(w_bar, (w_pert - w)/eps_fd)
    
    V_bar = zero(V)
    CloudSBP.solve_min_norm_rev!(V_bar, w_bar, V, b)
    dot_rev = sum(V_bar .* V_pert)
    
    @test isapprox(dot_rev, dot_fd, atol=1e-5)
end

@testset "test build_interpolation!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 1:4

    num_nodes = binomial(Dim + degree + 1, Dim)
    x = randn(Dim, num_nodes)
    num_interp = 2*num_nodes
    x_interp = randn(Dim, num_interp)

    lower = vec(minimum([x x_interp], dims=2))
    upper = vec(maximum([x x_interp], dims=2))
    xref = 0.5*(upper + lower)
    dx = 1.001*(upper - lower)

    R = zeros(num_interp, num_nodes)
    CloudSBP.build_interpolation!(R, degree, x, x_interp, xref, dx)

    # check that monomials are interpolated correctly
    num_basis = binomial(Dim + degree, Dim)
    V = zeros(num_nodes, num_basis)
    CloudSBP.monomial_basis!(V, degree, x, Val(Dim))
    RV = R*V
    V_interp = zeros(num_interp, num_basis)
    CloudSBP.monomial_basis!(V_interp, degree, x_interp, Val(Dim)) 
    for I in CartesianIndices(V_interp)
        @test isapprox(V_interp[I], RV[I], atol=1e-10)
    end 
end