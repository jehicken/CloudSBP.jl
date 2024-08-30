# Tests related to the Face struct

@testset "test build_face: dimension $Dim" for Dim in 1:3 
    for d = 1:Dim
        c1 = Cell(SVector(ntuple(i -> 0.0, Dim)),
                  SVector(ntuple(i -> 1.0, Dim)))
        c2 = Cell(SVector(ntuple(i -> i == d ? 1.0 : 0.0, Dim)),
                  SVector(ntuple(i -> 1.0, Dim)))
        face = CloudSBP.build_face(d, c1, c2)
        @test face.dir == d 
        @test isapprox(face.boundary.widths[d], 0.0)
        @test isapprox(face.boundary.origin[d], c2.boundary.origin[d])
        for it = 1:Dim-1
            i = mod(d+it-1,Dim) + 1
            @test isapprox(face.boundary.origin[i], c1.boundary.origin[i])
            @test isapprox(face.boundary.origin[i], c2.boundary.origin[i])
            @test isapprox(face.boundary.widths[i], c1.boundary.widths[i])
            @test isapprox(face.boundary.widths[i], c2.boundary.widths[i])
        end 
    end
end

@testset "test build_boundary_face: dimension $Dim" for Dim in 1:3 
    for d in [i for i=-Dim:Dim if i != 0]
        c = Cell(SVector(ntuple(i -> 0.0, Dim)), SVector(ntuple(i -> 1.0, Dim)))
        face = CloudSBP.build_boundary_face(d, c)
        @test face.dir == d 
        @test isapprox(face.boundary.widths[abs(d)], 0.0)
        if d > 0 
            @test isapprox(face.boundary.origin[d], c.boundary.origin[d] + 
                           c.boundary.widths[d])
        else 
            @test isapprox(face.boundary.origin[abs(d)],
                           c.boundary.origin[abs(d)])
        end
        for it = 1:Dim-1
            i = mod(abs(d)+it-1,Dim) + 1
            @test isapprox(face.boundary.origin[i], c.boundary.origin[i])
            @test isapprox(face.boundary.widths[i], c.boundary.widths[i])
        end
    end
end