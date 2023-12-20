# Tests related to diagonal norm construction 

@testset "test solve_min_norm!" begin
    num_eq = 10
    num_dof = 50
    V = randn(num_dof,num_eq)
    w = zeros(num_dof)
    b = randn(num_eq)
    
    eps_fd = 1e-6
    CutDGD.solve_min_norm!(w, V, b)
    V_pert = randn(size(V))
    w_pert = zero(w)
    CutDGD.solve_min_norm!(w_pert, V + eps_fd*V_pert, b)
    w_bar = randn(size(w))
    dot_fd = dot(w_bar, (w_pert - w)/eps_fd)
    
    V_bar = zero(V)
    CutDGD.solve_min_norm_rev!(V_bar, w_bar, V, b)
    dot_rev = sum(V_bar .* V_pert)
    
    @test isapprox(dot_rev, dot_fd, atol=1e-5)
end

@testset "test calc_moments: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> i == 1 ? 2.0 : 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # define a level-set that cuts the HyperRectangle in half 
    num_basis = 1
    xc = 0.5*ones(Dim, num_basis)
    xc[1, 1] = 1.0
    nrm = zeros(Dim, num_basis)
    nrm[1, 1] = 1.0
    tang = zeros(Dim, Dim-1, num_basis)
    tang[:, :, 1] = nullspace(reshape(nrm[:, 1], 1, Dim))
    crv = zeros(Dim-1, num_basis)
    rho = 100.0*num_basis    
    levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

    # Generate some DGD dof locations used to refine the background mesh
    num_nodes = 10*binomial(Dim + degree, Dim)
    points = rand(Dim, num_nodes)
    CutDGD.refine_on_points!(root, points)
    CutDGD.mark_cut_cells!(root, levset)

    num_cells = num_leaves(root)
    cell_xavg = zeros(Dim, num_cells)
    cell_dx = zero(cell_xavg)
    for (c,cell) in enumerate(allleaves(root))
        cell_xavg[:,c] = center(cell)
        cell_xavg[:,c] = 2*cell.boundary.widths
    end
    m = calc_moments(root, levset, degree, cell_xavg, cell_dx)
    
    # check that (scaled) zero-order moments sum to cut-domain volume
    

end

@testset "test cell_quadrature: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = num_basis
    x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}())) #Face{2,Float64}}()))
    CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
    xc = randn(Dim, num_nodes) .+ 0.5
    w = CutDGD.cell_quadrature(degree, xc, xq, wq, Val(Dim))

    V = zeros(num_nodes, num_basis)
    CutDGD.monomial_basis!(V, degree, xc, Val(Dim))
    Vq = zeros(num_quad, num_basis)
    CutDGD.monomial_basis!(Vq, degree, xq, Val(Dim))
    for k = 1:num_basis 
        integral_ref = dot(Vq[:,k], wq)
        integral = dot(V[:,k], w)
        @test isapprox(integral, integral_ref)
    end 
end 

@testset "test cell_quadrature_rev!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = 2*num_basis
    x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}())) #Face{2,Float64}}()))
    CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
    xc = randn(Dim, num_nodes) .+ 0.5

    # compute the derivative of the (weighted) quad weights w.r.t. xc 
    w_bar = randn(num_nodes)
    xc_bar = zero(xc)
    CutDGD.cell_quadrature_rev!(xc_bar, degree, xc, xq, wq, w_bar, Val(Dim))
    p = randn(size(xc_bar))
    dot_prod = dot(vec(xc_bar), vec(p))

    # now use complex step to approximate the same derivatives
    ceps = 1e-60
    xc_cmplx = complex.(xc, ceps*p)
    w_cmplx = CutDGD.cell_quadrature(degree, xc_cmplx, xq, wq, Val(Dim))
    dot_prod_cmplx = dot(w_bar, imag.(w_cmplx)/ceps)

    @test isapprox(dot_prod, dot_prod_cmplx)
end

@testset "test diagonal_norm: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:1

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # DGD dof locations
    num_basis = binomial(Dim + degree, Dim)

    num_nodes = 10*num_basis
    points = rand(Dim, num_nodes)

    # num1d = 3*num_basis
    # num_nodes = num1d^Dim
    # points = zeros(Dim, num_nodes)
    # xc = reshape(points, (Dim, ntuple(i -> num1d, Dim)...))
    # for I in CartesianIndices(xc)
    #     # I[1] is the coordinate, so I[I[1] + 1] is the index for that coord
    #     xc[I] = (I[I[1] + 1] - 1)/(num1d-1)
    # end

    # dx = 1/(num1d-1)
    # points .+= 0.2*dx*randn(Dim, num_nodes)
    
    # refine mesh, build stencil, and evaluate norm
    CutDGD.refine_on_points!(root, points)
    CutDGD.build_nn_stencils!(root, points, degree)
    H = CutDGD.diagonal_norm(root, points, degree) 

    count = 0
    for i = 1:num_nodes 
        if H[i] < 0.0
            #println("Negative weight found: ",H[i])
            count += 1
        end
    end
    println(count,"/",num_nodes," negative weights found!!!!!!")

    x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    CutDGD.quadrature!(xq, wq, root.boundary, x1d, w1d)
    
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
    num_nodes = 10*num_basis
    points = rand(Dim, num_nodes)
    
    # refine mesh, build sentencil, and evaluate norm
    CutDGD.refine_on_points!(root, points)
    CutDGD.build_nn_stencils!(root, points, degree)

    # get the derivative of dot(H_bar, H) in direction p
    H_bar = randn(num_nodes)
    points_bar = zero(points)
    CutDGD.diagonal_norm_rev!(points_bar, root, points, degree, H_bar)
    p = randn(Dim, num_nodes)
    dot_prod = dot(vec(points_bar), vec(p))

    # get the derivative using complex step
    ceps = 1e-60
    points_cmplx = complex.(points, ceps*p)
    H_cmplx = CutDGD.diagonal_norm(root, points_cmplx, degree) 
    dot_prod_cmplx = dot(H_bar, imag.(H_cmplx)/ceps)

    @test isapprox(dot_prod, dot_prod_cmplx)
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
    CutDGD.refine_on_points!(root, points)
    CutDGD.build_nn_stencils!(root, points, degree)
    Z, wp, num_var = CutDGD.get_null_and_part(root, points, degree)    

    # compute objective gradient for random y 
    rho = 1.0
    y = 0.1*randn(num_var)
    g = zero(y)
    CutDGD.obj_norm_grad!(g, root, wp, Z, y, rho, num_nodes)

    # compare against a complex-step based directional derivative 
    p = randn(num_var)
    gdotp = dot(g, p)

    ceps = 1e-60
    num_cell = CutDGD.num_leaves(root)
    wp_cmplx = Vector{Vector{ComplexF64}}(undef, num_cell)
    Z_cmplx = Vector{Matrix{ComplexF64}}(undef, num_cell)
    for i = 1:num_cell 
        wp_cmplx[i] = complex.(wp[i])
        Z_cmplx[i] = complex.(Z[i])
    end
    y_cmplx = complex.(y, ceps.*p)
    rho_cmplx = complex(rho)
    obj = CutDGD.obj_norm(root, wp_cmplx, Z_cmplx, y_cmplx, rho_cmplx,
                          num_nodes)
    gdotp_cmplx = imag(obj)/ceps 

    @test isapprox(gdotp, gdotp_cmplx)

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
    CutDGD.refine_on_points!(root, points)
    CutDGD.build_nn_stencils!(root, points, degree)

    # compute the penalty gradient 
    mu = 1.0
    g = zeros(Dim*num_nodes)
    dist_ref = 1.0 .+ 0.1*rand(num_nodes)
    CutDGD.penalty_grad!(g, root, points, points_init, dist_ref, mu, degree)

    # compare against a complex-step based directional derivative 
    p = randn(length(g))
    gdotp = dot(g, p)

    ceps = 1e-60
    points_cmplx = complex.(points, ceps.*reshape(p, size(points)))
    penalty = CutDGD.penalty(root, points_cmplx, points_init, dist_ref, mu, degree)
    gdotp_cmplx = imag(penalty)/ceps 

    @test isapprox(gdotp, gdotp_cmplx)
end