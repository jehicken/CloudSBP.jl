# Tests related to diagonal norm construction 

@testset "test calc_moments: dimension $Dim, degree $degree" for Dim in 2:3, degree in 1:4

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)), 
                CellData(Vector{Int}(), Vector{Int}()))

    # define a level-set that cuts the HyperRectangle
    levset(x) = x[1] - 1/pi

    # Generate some DGD dof locations used to refine the background mesh
    num_nodes = 5*binomial(Dim + degree, Dim)
    points = rand(Dim, num_nodes)
    CloudSBP.refine_on_points!(root, points)
    CloudSBP.mark_cut_cells!(root, levset)
    CloudSBP.set_xref_and_dx!(root, points)

    # evaluate the moments and ...
    m = CloudSBP.calc_moments!(root, levset, 2*degree-1, degree)

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
    R = 1/3
    levset(x) = norm(x .- 0.5)^2 - R^2

    # Generate some DGD dof locations used to refine the background mesh
    num_nodes = 10*binomial(Dim + degree, Dim)
    points = rand(Dim, num_nodes)
    CloudSBP.refine_on_points!(root, points)
    for cell in allleaves(root)
        split!(cell, CloudSBP.get_data)
    end
    CloudSBP.mark_cut_cells!(root, levset)
    CloudSBP.set_xref_and_dx!(root, points)

    # evaluate the moments and ...
    fit_degree = 2
    m = CloudSBP.calc_moments!(root, levset, max(fit_degree,2*degree-1),
                             fit_degree) 

    # ...check that (scaled) zero-order moments sum to cut-domain volume scaled
    # by the constant basis
    vol = 0.0
    for (c,cell) in enumerate(allleaves(root))
        vol += m[1,c]
    end
    tet_vol = 2^Dim/factorial(Dim)
    basis_val = 1/sqrt(tet_vol)
    vol /= basis_val

    @test isapprox(vol, 1 - sphere_vol(R, Val(Dim)) , atol=1e-4)
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
    m = CloudSBP.calc_moments!(cell, degree)
    xc = randn(Dim, num_nodes) .+ 0.5
    w = CloudSBP.cell_quadrature(degree, xc, m, cell.data.xref, cell.data.dx,
                               Val(Dim))

    # Check the cell quadrature against the moments computed using LG quad.
    x1d, w1d = CloudSBP.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    CloudSBP.quadrature!(xq, wq, cell.boundary, x1d, w1d)
    V = zeros(num_nodes, num_basis)
    CloudSBP.monomial_basis!(V, degree, xc, Val(Dim))
    Vq = zeros(num_quad, num_basis)
    CloudSBP.monomial_basis!(Vq, degree, xq, Val(Dim))
    for k = 1:num_basis 
        integral_ref = dot(Vq[:,k], wq)
        integral = dot(V[:,k], w)
        @test isapprox(integral, integral_ref)
    end 
end

@testset "test cell_quadrature (old): dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = num_basis
    x1d, w1d = CloudSBP.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}())) #Face{2,Float64}}()))
    CloudSBP.quadrature!(xq, wq, cell.boundary, x1d, w1d)
    xc = randn(Dim, num_nodes) .+ 0.5
    w = CloudSBP.cell_quadrature(degree, xc, xq, wq, Val(Dim))

    V = zeros(num_nodes, num_basis)
    CloudSBP.monomial_basis!(V, degree, xc, Val(Dim))
    Vq = zeros(num_quad, num_basis)
    CloudSBP.monomial_basis!(Vq, degree, xq, Val(Dim))
    for k = 1:num_basis 
        integral_ref = dot(Vq[:,k], wq)
        integral = dot(V[:,k], w)
        @test isapprox(integral, integral_ref)
    end 
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
    CloudSBP.refine_on_points!(root, points)
    CloudSBP.build_nn_stencils!(root, points, degree)
    CloudSBP.set_xref_and_dx!(root, points)
    m = CloudSBP.calc_moments!(root, degree)
    H = zeros(num_nodes)
    CloudSBP.diagonal_norm!(H, root, points, degree)

    # get quadrature to compute reference integrals
    x1d, w1d = CloudSBP.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    CloudSBP.quadrature!(xq, wq, root.boundary, x1d, w1d)
    
    # compare H-based quadrature with LG-based quadrature
    V = zeros(num_nodes, num_basis)
    CloudSBP.monomial_basis!(V, degree, points, Val(Dim))
    Vq = zeros(num_quad, num_basis)
    CloudSBP.monomial_basis!(Vq, degree, xq, Val(Dim))
    for k = 1:num_basis
        integral_ref = dot(Vq[:,k], wq)
        integral = dot(V[:,k], H)
        @test isapprox(integral, integral_ref)
    end 

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
    CloudSBP.refine_on_points!(root, points)
    CloudSBP.build_nn_stencils!(root, points, degree)
    CloudSBP.set_xref_and_dx!(root, points)
    m = CloudSBP.calc_moments!(root, degree)
    wp, Z, num_var = CloudSBP.get_null_and_part(root, points, degree)
    # compute a random contribution to the minimum norm solution
    y = 0.1*randn(num_var)
    ptr = 0
    for (c, cell) in enumerate(allleaves(root))
        num_dof = size(Z[c],2)
        cell.data.wts = wp[c] + Z[c]*y[ptr+1:ptr+num_dof]
        ptr += num_dof
    end
    H = zeros(num_nodes)
    CloudSBP.diagonal_norm!(H, root)

    # get quadrature to compute reference integrals
    x1d, w1d = CloudSBP.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(x1d)^Dim
    xq = zeros(Dim, num_quad)
    wq = zeros(num_quad)
    CloudSBP.quadrature!(xq, wq, root.boundary, x1d, w1d)
    
    # compare H-based quadrature with LG-based quadrature
    V = zeros(num_nodes, num_basis)
    CloudSBP.monomial_basis!(V, degree, points, Val(Dim))
    Vq = zeros(num_quad, num_basis)
    CloudSBP.monomial_basis!(Vq, degree, xq, Val(Dim))
    for k = 1:num_basis
        integral_ref = dot(Vq[:,k], wq)
        integral = dot(V[:,k], H)
        @test isapprox(integral, integral_ref)
    end
end