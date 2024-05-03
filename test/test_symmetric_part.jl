
@testset "test cell_symmetric_part: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4

    # use a unit HyperRectangle centered at the origin
    cell = Cell(SVector(ntuple(i -> -0.5, Dim)),
                SVector(ntuple(i -> 1.0, Dim)),
                CellData(Vector{Int}(), Vector{Int}()))

    # create a point cloud 
    num_basis = binomial(Dim + degree, Dim)
    num_nodes = num_basis + 1
    xc = randn(Dim, num_nodes)
    CutDGD.set_xref_and_dx!(cell, xc)

    # get the boundary operator 
    E = CutDGD.cell_symmetric_part(cell, xc, degree)

    # get quadrature points for the cell; these are used to integrate derivatives of the 
    # boundary integrands using the divergence theorem.
    x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(w1d)^Dim
    wq = zeros(num_quad)
    xq = zeros(Dim, num_quad)
    CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)

    # evaluate the monomial basis at the point cloud and quadrature points 
    V = zeros(num_nodes, num_basis)
    CutDGD.monomial_basis!(V, degree, xc, Val(Dim))
    Vq = zeros(num_quad, num_basis)
    CutDGD.monomial_basis!(Vq, degree, xq, Val(Dim))
    dVq = zeros(num_quad, num_basis, Dim)
    CutDGD.monomial_basis_derivatives!(dVq, degree, xq, Val(Dim))

    # multiply by random vector to reduce the number of tests 
    lvec = randn(num_basis)
    rvec = randn(num_basis)

    for di = 1:Dim
        VtEV = V'*E[:,:,di]*V 
        VtHdV = Vq'*(wq.*dVq[:,:,di]) + dVq[:,:,di]'*(wq.*Vq)
        @test isapprox( dot(lvec, VtEV*rvec), dot(lvec, VtHdV*rvec), atol=1e-12)
    end
end

@testset "test cell_symmetric_part (cut cell version): degree $degree" for degree in 1:4

    @testset "dimension = 1" begin
        Dim = 1
        # use a unit HyperRectangle centered at the origin
        cell = Cell(SVector{1}([-0.5]), SVector{1}([1.0]),
                    CellData(Vector{Int}(), Vector{Int}()))
        levset = x -> (x[1] + 0.5)^2 - 0.25

        # create a point cloud 
        num_basis = binomial(Dim + degree, Dim)
        num_nodes = binomial(Dim + degree + 1, Dim)
        xc = randn(Dim, num_nodes)
        CutDGD.set_xref_and_dx!(cell, xc)

        # get the boundary operator 
        E = CutDGD.cell_symmetric_part(cell, xc, degree, levset)

        # get quadrature points for the cell using Saye's algorithm directly;
        # these are used to integrate derivatives of the boundary integrands
        # using the divergence theorem.
        wq, xq = cut_cell_quad(cell.boundary, levset, degree+1, fit_degree=degree)
        num_quad = length(wq)

        # evaluate the monomial basis at the point cloud and quadrature points 
        V = zeros(num_nodes, num_basis)
        CutDGD.monomial_basis!(V, degree, xc, Val(Dim))
        Vq = zeros(num_quad, num_basis)
        CutDGD.monomial_basis!(Vq, degree, xq, Val(Dim))
        dVq = zeros(num_quad, num_basis, Dim)
        CutDGD.monomial_basis_derivatives!(dVq, degree, xq, Val(Dim))

        for di = 1:Dim
            for p = 1:num_basis
                for q = 1:num_basis
                    integral = vec(V[:,p])'*E[:,:,di]*vec(V[:,q])
                    ref_value = dot(wq, dVq[:,p,di].*Vq[:,q] + dVq[:,q,di].*Vq[:,p])
                    @test isapprox(integral, ref_value, atol=1e-10)
                end
            end
        end
    end 

    @testset "dimension = 2" begin 
        # This test works by setting the level-set to be a degree `j <= degree`
        # polynomial, and then using an iterated integral to evalute the
        # integrand `f'(x) g(y) + f(x) g(y)' = f'(x)` where `g(y) = 1`.  This
        # allows Saye's quadrature to exactly integrate the integrand.

        Dim = 2
        cell = Cell(SVector{2}([-1.0, -1.0]), SVector{2}([2.0, 2.0]),
                    CellData(Vector{Int}(), Vector{Int}()))

        # create a point cloud 
        num_basis = binomial(Dim + degree, Dim)
        num_nodes = binomial(Dim + degree + 1, Dim)
        xc = randn(Dim, num_nodes)
        CutDGD.set_xref_and_dx!(cell, xc)
        work = zeros(2)
        lval = zeros(1)
        xval = zeros(1)
        
        for j = 0:degree
            i = degree - j

            # define the degree j level-set 
            xval[1] = 1.0
            CutDGD.jacobi_poly!(lval, xval, 0.0, 0.0, j, work)
            fac = 1/lval[1]
            function levset(x)
                xval[1] = x[1]
                CutDGD.jacobi_poly!(lval, xval, 0.0, 0.0, j, work)
                return lval[1]*fac - x[2]
            end 

            # get the boundary operator 
            E = CutDGD.cell_symmetric_part(cell, xc, degree, levset)

            # get quadrature points for the cell using Saye's algorithm directly;
            # these are used to integrate derivatives of the boundary integrands
            # using the divergence theorem.
            wq, xq = cut_cell_quad(cell.boundary, levset, degree+1, fit_degree=degree)
            num_quad = length(wq)

            # define the function begin integrated 
            V = zeros(num_nodes)
            Vwork = zeros(2*num_nodes)
            CutDGD.jacobi_poly!(V, view(xc,1,:), 0.0, 0.0, i, Vwork)
            dVq = CutDGD.diff_jacobi_poly(view(xq,1,:), 0.0, 0.0, i)

            integral = ones(num_nodes)'*E[:,:,1]*V
            ref_value = dot(wq, dVq)
            @test isapprox(integral, ref_value, atol=1e-10)
        end
    end

    @testset "dimension = 3" begin 
        # This test works by setting the level-set to be a degree `j <= degree`
        # polynomial, and then using an iterated integral to evalute the
        # integrand `f'(x) g(y) + f(x) g(y)' = f'(x)` where `g(y) = 1`.  This
        # allows Saye's quadrature to exactly integrate the integrand.

        Dim = 3
        cell = Cell(SVector{3}([-1.0, -1.0, -1.0]), SVector{3}([2.0, 2.0, 2.1]),
                    CellData(Vector{Int}(), Vector{Int}()))

        # create a point cloud 
        num_basis = binomial(Dim + degree, Dim)
        num_nodes = binomial(Dim + degree + 1, Dim)
        xc = randn(Dim, num_nodes)
        CutDGD.set_xref_and_dx!(cell, xc)
        work = zeros(3)
        lval = zeros(1)
        xval = zeros(2)
        
        for i = 0:degree 
            for j = 0:degree

                # define the degree `2*degree -i - j` level set 
                xval[:] = [1.0; 1.0]
                CutDGD.jacobi_poly!(lval, view(xval,1:1), 0.0, 0.0, degree-i, work)
                lx = lval[1] 
                CutDGD.jacobi_poly!(lval, view(xval,2:2), 0.0, 0.0, degree-j, work)
                fac = 1/(lx*lval[1])
                function levset(x)
                    CutDGD.jacobi_poly!(lval, view(x,1:1), 0.0, 0.0, degree-i, work)
                    lx = lval[1] 
                    CutDGD.jacobi_poly!(lval, view(x,2:2), 0.0, 0.0, degree-j, work)
                    return lx*lval[1]*fac - x[3]
                end

                # get the boundary operator
                E = CutDGD.cell_symmetric_part(cell, xc, degree, levset)

                # get quadrature points for the cell using Saye's algorithm directly;
                # these are used to integrate derivatives of the boundary integrands
                # using the divergence theorem.
                wq, xq = cut_cell_quad(cell.boundary, levset, degree+1, fit_degree=degree)
                num_quad = length(wq)

                # define the degree `i+j` function being integrated
                Vx = zeros(num_nodes)
                Vy = zero(Vx)
                Vwork = zeros(3*num_nodes)
                CutDGD.jacobi_poly!(Vx, view(xc,1,:), 0.0, 0.0, i, Vwork)
                CutDGD.jacobi_poly!(Vy, view(xc,2,:), 0.0, 0.0, j, Vwork)

                Vqx = zeros(num_quad)
                Vqy = zero(Vqx)
                Vqwork = zeros(3*num_quad)
                dVqx = zero(Vqx)
                dVqy = zero(Vqx)
                CutDGD.jacobi_poly!(Vqx, view(xq,1,:), 0.0, 0.0, i, Vqwork)
                CutDGD.jacobi_poly!(Vqy, view(xq,2,:), 0.0, 0.0, j, Vqwork)
                dVqx = CutDGD.diff_jacobi_poly(view(xq,1,:), 0.0, 0.0, i)
                dVqy = CutDGD.diff_jacobi_poly(view(xq,2,:), 0.0, 0.0, j)
                dVq = zeros(num_quad,2) 
                dVq[:,1] = dVqx.*Vqy 
                dVq[:,2] = Vqx.*dVqy 

                integral = Vx'*E[:,:,1]*Vy
                ref_value = dot(wq, dVq[:,1])
                @test isapprox(integral, ref_value, atol=1e-10)

                integral = Vx'*E[:,:,2]*Vy 
                ref_value = dot(wq, dVq[:,2])
                @test isapprox(integral, ref_value, atol=1e-10)
            end
        end
    end 

end

@testset "test make_compatible!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 1:4
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
    
    # get the quadrature (norm) and symmetric part
    m = CutDGD.calc_moments!(cell, levset, 2*degree-1, min(3,degree))
    w = CutDGD.cell_quadrature(2*degree-1, xc, m, cell.data.xref, cell.data.dx, Val(Dim))    
    E = CutDGD.cell_symmetric_part(cell, xc, degree, levset, fit_degree=min(3,degree))

    # modify E to make it compatible with H
    CutDGD.make_compatible!(E, w, cell, xc, degree)

    # check for compatibility using monomial basis
    V = zeros(num_nodes, num_basis)
    CutDGD.monomial_basis!(V, degree, xc, Val(Dim))
    dV = zeros(num_nodes, num_basis, Dim)
    CutDGD.monomial_basis_derivatives!(dV, degree, xc, Val(Dim))

    # multiply by random vector to reduce the number of tests 
    lvec = randn(num_basis)
    rvec = randn(num_basis)
    
    for d = 1:Dim
        VtEV = V'*E[:,:,d]*V
        VtHdV = V'*(w.*dV[:,:,d]) + dV[:,:,d]'*(w.*V)
        @test isapprox(dot(lvec, VtEV*rvec), dot(lvec, VtHdV*rvec), atol=1e-10)
    end
end