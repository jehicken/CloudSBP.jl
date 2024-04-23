# Tests related to diagonal norm construction 

@testset "test calc_moments: dimension $Dim, degree $degree" for Dim in 2:3, degree in 1:4

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)), 
                CellData(Vector{Int}(), Vector{Int}()))

    # define a level-set that cuts the HyperRectangle
    num_basis = 1
    xc = 0.5*ones(Dim, num_basis)
    xc[1, 1] = 1/pi
    nrm = zeros(Dim, num_basis)
    nrm[1, 1] = 1.0 
    tang = zeros(Dim, Dim-1, num_basis)
    tang[:, :, 1] = nullspace(reshape(nrm[:, 1], 1, Dim))
    crv = zeros(Dim-1, num_basis)
    rho = 100.0*num_basis    
    levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)
    levset_func(x) = evallevelset(x, levset)

    # Generate some DGD dof locations used to refine the background mesh
    num_nodes = 5*binomial(Dim + degree, Dim)
    points = rand(Dim, num_nodes)
    CutDGD.refine_on_points!(root, points)
    CutDGD.mark_cut_cells!(root, levset)
    CutDGD.set_xref_and_dx!(root, points)

    # evaluate the moments and ...
    m = CutDGD.calc_moments!(root, levset_func, 2*degree-1, degree)

    # ...check that (scaled) zero-order moments sum to cut-domain volume scaled
    # by the constant basis
    vol = 0.0
    for (c,cell) in enumerate(allleaves(root))
        vol += m[1,c]
    end
    tet_vol = 2^Dim/factorial(Dim)
    basis_val = 1/sqrt(tet_vol)
    vol /= basis_val

    @test isapprox(vol, 1 - 1/pi, atol=1e-10)
end

# sphere_vol computes the volume of the Dim dimensional hypersphere
function sphere_vol(r, ::Val{Dim}) where {Dim}
    return 2*pi*r^2*sphere_vol(r, Val(Dim-2))/Dim
end
sphere_vol(r, ::Val{0}) = 1
sphere_vol(r, ::Val{1}) = 2*r

@testset "test calc_moments (sphere): dimension $Dim, degree $degree" for Dim in 2:3, degree in 1:4
    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)), 
                CellData(Vector{Int}(), Vector{Int}()))

    # define a level-set for a circle 
    num_basis = 8*4^(Dim-1)
    xc = randn(Dim, num_basis)
    nrm = zero(xc) 
    tang = zeros(Dim, Dim-1, num_basis)
    crv = zeros(Dim-1, num_basis)
    R = 1/3
    for i = 1:num_basis 
        #theta = 2*pi*(i-1)/(num_basis-1)
        #xc[:,i] = R*[cos(theta); sin(theta)] + [0.5;0.5]
        #nrm[:,i] = [cos(theta); sin(theta)]
        nrm[:,i] = xc[:,i]/norm(xc[:,i])
        xc[:,i] = nrm[:,i]*R + 0.5*ones(Dim)
        tang[:,:,i] = nullspace(reshape(nrm[:, i], 1, Dim))
        crv[:,i] .= 1/R
    end
    rho = 100.0*(num_basis)^(1/(Dim-1))
    levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)
    levset_func(x) = evallevelset(x, levset)

    # Generate some DGD dof locations used to refine the background mesh
    num_nodes = 10*binomial(Dim + degree, Dim)
    points = rand(Dim, num_nodes)
    CutDGD.refine_on_points!(root, points)
    for cell in allleaves(root)
        split!(cell, CutDGD.get_data)
    end
    CutDGD.mark_cut_cells!(root, levset)
    CutDGD.set_xref_and_dx!(root, points)

    # evaluate the moments and ...
    m = CutDGD.calc_moments!(root, levset_func, 2*degree-1, degree) 

    # ...check that (scaled) zero-order moments sum to cut-domain volume scaled
    # by the constant basis
    vol = 0.0
    for (c,cell) in enumerate(allleaves(root))
        vol += m[1,c]
    end
    tet_vol = 2^Dim/factorial(Dim)
    basis_val = 1/sqrt(tet_vol)
    vol /= basis_val

    @test isapprox(vol, 1 - sphere_vol(R, Val(Dim)) , atol=0.01)
end

@testset "test cell_quadrature: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = num_basis
    cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))
    num_cells = 1
    cell.data.xref = 0.5*ones(Dim)
    cell.data.dx = ones(Dim)
    m = CutDGD.calc_moments!(cell, degree)
    xc = randn(Dim, num_nodes) .+ 0.5
    w = CutDGD.cell_quadrature(degree, xc, m, cell.data.xref, cell.data.dx,
                               Val(Dim))

    # Check the cell quadrature against the moments computed using LG quad.
    x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
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
    num_nodes = num_basis
    cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))
    num_cells = 1
    cell.data.xref = 0.5*ones(Dim)
    cell.data.dx = ones(Dim)
    m = CutDGD.calc_moments!(cell, degree)
    xc = randn(Dim, num_nodes) .+ 0.5

    # compute the derivative of the (weighted) quad weights w.r.t. xc 
    w_bar = randn(num_nodes)
    xc_bar = zero(xc)
    CutDGD.cell_quadrature_rev!(xc_bar, degree, xc, m, cell.data.xref,
                                cell.data.dx, w_bar, Val(Dim))
    p = randn(size(xc_bar))
    dot_prod = dot(vec(xc_bar), vec(p))

    # now use complex step to approximate the same derivatives
    ceps = 1e-60
    xc_cmplx = complex.(xc, ceps*p)
    w_cmplx = CutDGD.cell_quadrature(degree, xc_cmplx, m, cell.data.xref,
                                     cell.data.dx, Val(Dim))
    dot_prod_cmplx = dot(w_bar, imag.(w_cmplx)/ceps)

    @test isapprox(dot_prod, dot_prod_cmplx)
end

@testset "test cell_quadrature (old): dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
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

@testset "test cell_quadrature_rev! (old): dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
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
    H = zeros(num_nodes)
    CutDGD.diagonal_norm!(H, root, points, degree)

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

    # get the derivative of dot(H_bar, H) in direction p
    H_bar = randn(num_nodes)
    points_bar = zero(points)
    CutDGD.diagonal_norm_rev!(points_bar, H_bar, root, points, degree)
    p = randn(Dim, num_nodes)
    dot_prod = dot(vec(points_bar), vec(p))

    # get the derivative using complex step
    ceps = 1e-60
    points_cmplx = complex.(points, ceps*p)
    H_cmplx = zeros(ComplexF64, num_nodes)
    CutDGD.diagonal_norm!(H_cmplx, root, points_cmplx, degree) 
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
    CutDGD.set_xref_and_dx!(root, points)
    m = CutDGD.calc_moments!(root, degree)

    # compute the penalty gradient 
    mu = 1.0
    H_tol = 1e-3.*ones(num_nodes)
    g = zeros(Dim*num_nodes)
    dist_ref = 1.0 .+ 0.1*rand(num_nodes)
    CutDGD.penalty_grad!(g, root, points, points_init, dist_ref, H_tol, mu,
                         degree)

    # compare against a complex-step based directional derivative 
    p = randn(length(g))
    gdotp = dot(g, p)

    ceps = 1e-60
    points_cmplx = complex.(points, ceps.*reshape(p, size(points)))
    penalty = CutDGD.penalty(root, points_cmplx, points_init, dist_ref, H_tol,
                             mu, degree)
    gdotp_cmplx = imag(penalty)/ceps 

    @test isapprox(gdotp, gdotp_cmplx)
end