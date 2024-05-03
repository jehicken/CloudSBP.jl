
@testset "test cell_skew_part: dimension $Dim, degree $degree" for Dim in 1:3, degree in 1:4

    @testset "uncut cell" begin
        # use a unit HyperRectangle centered at the origin
        cell = Cell(SVector(ntuple(i -> -0.5, Dim)),
                    SVector(ntuple(i -> 1.0, Dim)),
                    CellData(Vector{Int}(), Vector{Int}()))

        # create a point cloud 
        num_basis = binomial(Dim + degree, Dim)
        num_nodes = binomial(Dim + 2*degree -1, Dim)
        xc = randn(Dim, num_nodes)
        cell.data.points = 1:num_nodes
        CutDGD.set_xref_and_dx!(cell, xc)

        # get the quadrature
        m = CutDGD.calc_moments!(cell, 2*degree-1)
        w = CutDGD.cell_quadrature(2*degree-1, xc, m, cell.data.xref, cell.data.dx, Val(Dim))

        # get the symmetric boundary operator 
        E = CutDGD.cell_symmetric_part(cell, xc, degree)

        # get the skew-symmetric operator
        S = CutDGD.cell_skew_part(E, w, cell, xc, degree)

        # form D and check that it exactly differentiates polynomials up to degree
        V = zeros(num_nodes, num_basis)
        CutDGD.monomial_basis!(V, degree, xc, Val(Dim))
        dV = zeros(num_nodes, num_basis, Dim)
        CutDGD.monomial_basis_derivatives!(dV, degree, xc, Val(Dim))
        
        # left and right multiply by random vectors to reduce the number of tests 
        lvec = randn(num_nodes)
        rvec = randn(num_basis)

        for d = 1:Dim
            QV = S[:,:,d]*V + 0.5*E[:,:,d]*V
            HdV = diagm(w)*dV[:,:,d]
            @test isapprox(dot(lvec, QV*rvec), dot(lvec, HdV*rvec), atol=1e-10)

            #for i = 1:num_nodes 
            #    for j = 1:num_basis 
            #        @test isapprox(QV[i,j], HdV[i,j], atol=1e-10)
            #    end
            #end
        end 
    end

    @testset "cut cell" begin
        # use a unit HyperRectangle centered at the origin
        cell = Cell(SVector(ntuple(i -> -0.5, Dim)),
                    SVector(ntuple(i -> 1.0, Dim)),
                    CellData(Vector{Int}(), Vector{Int}()))

        if Dim == 1
            levset = x -> (x[1] + 0.5)^2 - 0.25
        elseif Dim == 2
            levset = x -> 4*(x[1] + 1.5)^2 + 36*x[2]^2 - 9
        elseif Dim == 3
            levset = x -> (x[1] + 0.5)^2 + x[2]^2 + x[3]^2 - 0.25^2
        end

        # create a point cloud 
        num_basis = binomial(Dim + degree, Dim)
        num_nodes = binomial(Dim + 2*degree -1, Dim)
        xc = randn(Dim, num_nodes)
        cell.data.points = 1:num_nodes
        CutDGD.set_xref_and_dx!(cell, xc)
        cell.data.cut = true

        # get the quadrature
        m = CutDGD.calc_moments!(cell, levset, 2*degree-1, min(3,degree))
        w = CutDGD.cell_quadrature(2*degree-1, xc, m, cell.data.xref, cell.data.dx, Val(Dim))

        # get the symmetric boundary operator 
        E = CutDGD.cell_symmetric_part(cell, xc, degree, levset, fit_degree=min(3,degree))
        CutDGD.make_compatible!(E, w, cell, xc, degree)

        # get the skew-symmetric operator
        S = CutDGD.cell_skew_part(E, w, cell, xc, degree)

        # form D and check that it exactly differentiates polynomials up to degree
        V = zeros(num_nodes, num_basis)
        CutDGD.monomial_basis!(V, degree, xc, Val(Dim))
        dV = zeros(num_nodes, num_basis, Dim)
        CutDGD.monomial_basis_derivatives!(dV, degree, xc, Val(Dim))

        # left and right multiply by random vectors to reduce the number of tests 
        lvec = randn(num_nodes)
        rvec = randn(num_basis)

        for d = 1:Dim
            QV = S[:,:,d]*V + 0.5*E[:,:,d]*V
            HdV = diagm(w)*dV[:,:,d]
            @test isapprox(dot(lvec, QV*rvec), dot(lvec, HdV*rvec), atol=1e-10)


            #for i = 1:num_nodes
            #    for j = 1:num_basis
            #        @test isapprox(QV[i,j], HdV[i,j], atol=1e-10)
            #    end
            #end
        end
    end 
    
end

@testset "test interface_skew_part: dimension $Dim, degree $degree" for Dim in 1:3, degree in 1:4

    @testset "uncut cells" begin
        # use a pair of HyperRectangles on either side of origin
        cell_left = Cell(SVector(ntuple(i -> i == 1.0 ? -1.0 : 0.0, Dim)),
                         SVector(ntuple(i -> 1.0, Dim)),
                         CellData(Vector{Int}(), Vector{Int}()))
        cell_right = Cell(SVector(ntuple(i -> 0.0, Dim)),
                          SVector(ntuple(i -> 1.0, Dim)),
                          CellData(Vector{Int}(), Vector{Int}()))

        # add an interface
        dir = 1
        face = CutDGD.build_face(dir, cell_left, cell_right)
        
        # create a point cloud 
        num_basis = binomial(Dim + degree, Dim)
        num_nodes = 3*num_basis
        xc = randn(Dim, num_nodes)
        xc[1,:] .*= 2.0 # to spread out the cells
        cell_left.data.points = 1:num_basis
        cell_right.data.points = (num_basis+1):num_nodes
        CutDGD.set_xref_and_dx!(cell_left, xc)
        CutDGD.set_xref_and_dx!(cell_right, xc)

        # get the interface matrix 
        xc_left = view(xc, :, face.cell[1].data.points)
        xc_right = view(xc, :, face.cell[2].data.points)
        Sface = CutDGD.interface_skew_part(face, xc_left, xc_right, degree)

        # get the face quadrature for testing purposes
        x1d, w1d = CutDGD.lg_nodes(degree+1)
        wq_face = zeros(length(w1d)^(Dim-1))
        xq_face = zeros(Dim, length(wq_face))
        CutDGD.face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)

        # Evaluate the monomials at the point cloud and the face quadrature
        V_left = zeros(length(cell_left.data.points), num_basis)
        V_right = zeros(length(cell_right.data.points), num_basis)
        CutDGD.monomial_basis!(V_left, degree, xc_left, Val(Dim))
        CutDGD.monomial_basis!(V_right, degree, xc_right, Val(Dim))
        Vq = zeros(length(wq_face), num_basis)
        CutDGD.monomial_basis!(Vq, degree, xq_face, Val(Dim))
        
        # left and right multiply by random vectors to reduce the number of tests 
        lvec = randn(num_basis)
        rvec = randn(num_basis)

        VtSV = V_left'*Sface*V_right
        VtWV = 0.5*Vq'*(wq_face.*Vq) # need factor of 0.5, which is included in Sface

        @test isapprox(dot(lvec, VtSV*rvec), dot(lvec, VtWV*rvec), atol=1e-10)
 
    end

    @testset "cut cells" begin
        # use a pair of HyperRectangles on either side of origin
        cell_left = Cell(SVector(ntuple(i -> i == 1.0 ? -1.0 : 0.0, Dim)),
                         SVector(ntuple(i -> 1.0, Dim)),
                         CellData(Vector{Int}(), Vector{Int}()))
        cell_right = Cell(SVector(ntuple(i -> 0.0, Dim)),
                          SVector(ntuple(i -> 1.0, Dim)),
                          CellData(Vector{Int}(), Vector{Int}()))

        # add an interface
        dir = 1
        face = CutDGD.build_face(dir, cell_left, cell_right)

        if Dim == 1
            levset = x -> x[1]^2 - 0.25
        elseif Dim == 2
            levset = x -> x[1]^2 + x[2]^2 - 0.25
        elseif Dim == 3
            levset = x -> x[1]^2 + x[2]^2 + x[3]^2 - 0.25
        end

        # create a point cloud 
        num_basis = binomial(Dim + degree, Dim)
        num_nodes = 3*num_basis
        xc = randn(Dim, num_nodes)
        xc[1,:] .*= 2.0 # to spread out the cells
        cell_left.data.points = 1:num_basis
        cell_right.data.points = (num_basis+1):num_nodes
        CutDGD.set_xref_and_dx!(cell_left, xc)
        CutDGD.set_xref_and_dx!(cell_right, xc)
        cell_left.data.cut = true
        cell_right.data.cut = true 
        face.cut = true

        # get the interface matrix 
        xc_left = view(xc, :, face.cell[1].data.points)
        xc_right = view(xc, :, face.cell[2].data.points)
        Sface = CutDGD.interface_skew_part(face, xc_left, xc_right, degree,
                                           levset, fit_degree=min(3,degree))

        # get the face quadrature for testing purposes
        wq_face, xq_face = cut_face_quad(face.boundary, face.dir, levset, degree+1,
                                         fit_degree=min(3,degree))

        # Evaluate the monomials at the point cloud and the face quadrature
        V_left = zeros(length(cell_left.data.points), num_basis)
        V_right = zeros(length(cell_right.data.points), num_basis)
        CutDGD.monomial_basis!(V_left, degree, xc_left, Val(Dim))
        CutDGD.monomial_basis!(V_right, degree, xc_right, Val(Dim))
        Vq = zeros(length(wq_face), num_basis)
        CutDGD.monomial_basis!(Vq, degree, xq_face, Val(Dim))
        
        # left and right multiply by random vectors to reduce the number of tests 
        lvec = randn(num_basis)
        rvec = randn(num_basis)

        VtSV = V_left'*Sface*V_right
        VtWV = 0.5*Vq'*(wq_face.*Vq) # need factor of 0.5, which is included in Sface

        @test isapprox(dot(lvec, VtSV*rvec), dot(lvec, VtWV*rvec), atol=1e-10)

    end
end