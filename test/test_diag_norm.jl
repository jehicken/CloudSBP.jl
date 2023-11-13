# Tests related to diagonal norm construction 

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

@testset "test diagonal_norm: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4

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