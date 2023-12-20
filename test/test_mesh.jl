# Tests related to the mesh 

@testset "test is_cut: dimension $Dim" for Dim in 1:3

    # create a level approximation to the unit hypershpere 
    num_basis = 2*Dim
    xc = zeros(Dim, num_basis)
    nrm = zeros(Dim, num_basis)
    tang = zeros(Dim, Dim-1, num_basis)
    crv = -ones(Dim-1, num_basis)
    for d = 1:Dim 
        idx = (d-1)*2 + 1
        xc[d, idx] = -1.0
        nrm[d, idx] = -xc[d, idx]
        tang[:, :, idx] = nullspace(reshape(nrm[:,idx], 1, Dim))
        
        idx = (d-1)*2 + 2
        xc[d, idx] =  1.0 
        nrm[d, idx] = -xc[d, idx]
        tang[:, :, idx] = nullspace(reshape(nrm[:,idx], 1, Dim))
    end
    rho = 100.0*num_basis
    levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

    # make a rectangle inside hypersphere that is not cut 
    L = 0.99*sqrt(1/Dim)
    origin = SVector(ntuple(i -> -L, Dim))
    widths = SVector(ntuple(i -> 2*L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CutDGD.is_cut(rect, levset) == false

    # make a rectangle outside hypersphere that is not cut 
    L = 0.25
    origin = SVector(ntuple(i -> i == Dim ? 1.0001 : 0.0, Dim))
    widths = SVector(ntuple(i -> L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CutDGD.is_cut(rect, levset) == false

    # make a rectangle that is cut by the hypersphere
    L = 0.2 
    origin = SVector(ntuple(i -> i == Dim ? 1 - 0.5*L : 0.0, Dim))
    widths = SVector(ntuple(i -> L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CutDGD.is_cut(rect, levset) == true

end

@testset "test is_immersed: dimension $Dim" for Dim in 1:3

    # create a level approximation to the unit hypershpere 
    num_basis = 2*Dim
    xc = zeros(Dim, num_basis)
    nrm = zeros(Dim, num_basis)
    tang = zeros(Dim, Dim-1, num_basis)
    crv = -ones(Dim-1, num_basis)
    for d = 1:Dim 
        idx = (d-1)*2 + 1
        xc[d, idx] = -1.0
        nrm[d, idx] = -xc[d, idx]
        tang[:, :, idx] = nullspace(reshape(nrm[:,idx], 1, Dim))
        
        idx = (d-1)*2 + 2
        xc[d, idx] =  1.0 
        nrm[d, idx] = -xc[d, idx]
        tang[:, :, idx] = nullspace(reshape(nrm[:,idx], 1, Dim))
    end
    rho = 100.0*num_basis
    levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

    # make a rectangle inside hypersphere that is not cut 
    L = 0.99*sqrt(1/Dim)
    origin = SVector(ntuple(i -> -L, Dim))
    widths = SVector(ntuple(i -> 2*L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CutDGD.is_center_immersed(rect, levset) == true

    # make a rectangle outside hypersphere that is not cut 
    L = 0.25
    origin = SVector(ntuple(i -> i == Dim ? 1.0001 : 0.0, Dim))
    widths = SVector(ntuple(i -> L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CutDGD.is_center_immersed(rect, levset) == false

end