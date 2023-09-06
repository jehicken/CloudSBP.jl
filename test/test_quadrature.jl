# Tests related to cell and face quadratures

@testset "test quadrature!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:8
    x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
    xq = zeros(Dim, length(x1d)^Dim)
    wq = zeros(length(w1d)^Dim)
    cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
    SVector(ntuple(i -> 1.0, Dim)),
    CellData(Vector{Int}(), Vector{Int}())) #Face{2,Float64}}()))
    CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
    integral = 0.0
    for i = 1:size(xq,2)
        intq = 1.0
        for d = 1:Dim 
            intq *= xq[d,i]^(2*degree+1)
        end 
        integral += wq[i]*intq 
    end 
    @test isapprox(integral, 1/(2*degree+2)^(Dim))
end 

@testset "test face_quadrature!: dimension $Dim, degree $degree, dir. $dir" for Dim in 1:3, degree in 0:4, dir in 1:Dim
    x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
    xq = zeros(Dim, length(x1d)^(Dim-1))
    wq = zeros(length(w1d)^(Dim-1))
    face = Cell(SVector(ntuple(i -> 0.0, Dim)),
    SVector(ntuple(i -> i == dir ? 0.0 : 1.0, Dim)),
    CellData(Vector{Int}(), Vector{Int}())) #Vector{Face{2,Float64}}()))
    CutDGD.face_quadrature!(xq, wq, face.boundary, x1d, w1d, dir)
    integral = 0.0
    tangent_indices = ntuple(i -> i >= dir ? i+1 : i, Dim-1)
    for i = 1:size(xq,2)
        intq = 1.0
        for d in tangent_indices 
            intq *= xq[d,i]^(2*degree+1)
        end 
        integral += wq[i]*intq 
    end 
    @test isapprox(integral, 1/(2*degree+2)^(Dim-1))
end 