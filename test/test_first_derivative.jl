# Tests related to building the first-derivative operator 

@testset "test uncut_volume_integrate!: dimension $Dim" for Dim in 1:3

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # storage for sparse matrix information 
    rows = Array{Array{Int64}}(undef, Dim)
    cols = Array{Array{Int64}}(undef, Dim)
    Svals = Array{Array{Float64}}(undef, Dim)
    for d = 1:Dim
        rows[d] = zeros(Int, (0))
        cols[d] = zeros(Int, (0))
        Svals[d] = zeros(Float64, (0))
    end

    # DGD dof locations 
    degree = 2
    num_basis = binomial(Dim + degree, Dim)
    points = randn(Dim, num_basis)
    root.data.points = 1:num_basis

    CutDGD.uncut_volume_integrate!(rows, cols, Svals, root, points, degree)

    for d = 1:Dim 
        println("rows[",d,"] = ",rows[d])
        println("cols[",d,"] = ",cols[d])
        println("Svals[",d,"] = ",Svals[d])
    end
end 

@testset "test cut_volume_integrate!: dimension $Dim" for Dim in 2:3

    # use a unit HyperRectangle 
    root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # define a level-set that cuts the HyperRectangle into a Simplex 
    num_basis = Dim
    xc = zeros(Dim, num_basis)
    nrm = zeros(Dim, num_basis)
    tang = zeros(Dim, Dim-1, num_basis)
    crv = zeros(Dim-1, num_basis)
    for d = 1:Dim 
        idx = 1
        xc[d, idx] = 1.0
        nrm[:, idx] = ones(Dim)/sqrt(Dim)
        tang[:, :, idx] = nullspace(reshape(nrm[:,idx], 1, Dim))
    end
    rho = 100.0*num_basis
    levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

    # storage for sparse matrix information 
    rows = Array{Array{Int64}}(undef, Dim)
    cols = Array{Array{Int64}}(undef, Dim)
    Svals = Array{Array{Float64}}(undef, Dim)
    for d = 1:Dim
        rows[d] = zeros(Int, (0))
        cols[d] = zeros(Int, (0))
        Svals[d] = zeros(Float64, (0))
    end

    # DGD dof locations 
    degree = 2
    num_basis = binomial(Dim + degree, Dim)
    points = randn(Dim, num_basis)
    root.data.points = 1:num_basis

    #clevset = @safe_cfunction( x -> evallevelset(x, levset), Cdouble, 
    #                          (Vector{Float64},))
    #x = ones(Dim)
    #println("clevset(x) = ",clevset(x))

    CutDGD.mark_cut_cells!(root, levset)
    CutDGD.cut_volume_integrate!(rows, cols, Svals, root, levset, points, 
                                 degree)

    for d = 1:Dim 
        println("rows[",d,"] = ",rows[d])
        println("cols[",d,"] = ",cols[d])
        println("Svals[",d,"] = ",Svals[d])
    end
end 