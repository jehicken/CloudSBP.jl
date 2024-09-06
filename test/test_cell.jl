# Tests related to the Cells

@testset "test is_cut: dimension $Dim" for Dim in 1:3

    # create a level-set for the unit hypersphere
    levset(x) = norm(x)^2 - 1.0^2

    # make a rectangle inside hypersphere that is not cut 
    L = 0.99*sqrt(1/Dim)
    origin = SVector(ntuple(i -> -L, Dim))
    widths = SVector(ntuple(i -> 2*L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CloudSBP.is_cut(rect, levset) == false

    # make a rectangle outside hypersphere that is not cut 
    L = 0.25
    origin = SVector(ntuple(i -> i == Dim ? 1.0001 : 0.0, Dim))
    widths = SVector(ntuple(i -> L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CloudSBP.is_cut(rect, levset) == false

    # make a rectangle that is cut by the hypersphere
    L = 0.2 
    origin = SVector(ntuple(i -> i == Dim ? 1 - 0.5*L : 0.0, Dim))
    widths = SVector(ntuple(i -> L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CloudSBP.is_cut(rect, levset) == true

end

@testset "test is_immersed: dimension $Dim" for Dim in 1:3

    # create a level-set for the unit hypersphere
    levset(x) = norm(x)^2 - 1.0^2

    # make a rectangle inside hypersphere that is not cut 
    L = 0.99*sqrt(1/Dim)
    origin = SVector(ntuple(i -> -L, Dim))
    widths = SVector(ntuple(i -> 2*L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CloudSBP.is_center_immersed(rect, levset) == true

    # make a rectangle outside hypersphere that is not cut 
    L = 0.25
    origin = SVector(ntuple(i -> i == Dim ? 1.0001 : 0.0, Dim))
    widths = SVector(ntuple(i -> L, Dim))
    rect = HyperRectangle(origin, widths)
    @test CloudSBP.is_center_immersed(rect, levset) == false

end