# Tests related to building the first-derivative operator 

"""
    y = skew_matvec(S, x)

Multiplies the sparse matrix `S` by `x`, assuming that `S` stores the upper part
of a skew-symmetric matrix.
"""
function skew_matvec(S, x)
    y = zero(x)
    rows = rowvals(S)
    vals = nonzeros(S)
    m, n = size(S)
    for j = 1:n
        for i in nzrange(S, j)
            row = rows[i]
            val = vals[i]
            y[row] += val*x[j]
            y[j] -= val*x[row]
        end
    end
    return y 
end

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

    for di = 1:Dim
        for p = 1:num_basis
            for q = 1:num_basis
                integral = vec(V[:,p])'*E[:,:,di]*vec(V[:,q])
                ref_value = dot(wq, dVq[:,p,di].*Vq[:,q] + dVq[:,q,di].*Vq[:,p])
                @test isapprox(integral, ref_value, atol=1e-12)
            end
        end
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

# @testset "test uncut_volume_integrate!: dimension $Dim" for Dim in 1:3

#     # use a unit HyperRectangle 
#     root = Cell(SVector(ntuple(i -> 0.0, Dim)),
#                 SVector(ntuple(i -> 1.0, Dim)),
#                 CellData(Vector{Int}(), Vector{Int}()))

#     # storage for sparse matrix information 
#     rows = Array{Array{Int64}}(undef, Dim)
#     cols = Array{Array{Int64}}(undef, Dim)
#     Svals = Array{Array{Float64}}(undef, Dim)
#     for d = 1:Dim
#         rows[d] = Int[]
#         cols[d] = Int[]
#         Svals[d] = Float64[]
#     end

#     # for each polynomial degree `deg` from 1 to max_degree, we form all
#     # possible monomials u and v such that their total degree is `deg` or less.
#     # These monomials are evaluated at the random nodes `points`.  Then we
#     # contract these arrays with the skew symmetric SBP operator in each
#     # direction, and check for accuracy against the expected integral over a
#     # unit HyperRectangle.
#     max_degree = 4
#     for deg = 1:max_degree
#         num_basis = binomial(Dim + deg, Dim)
#         points = randn(Dim, num_basis) .+ 0.5*ones(Dim)
#         root.data.points = 1:num_basis        
#         CutDGD.uncut_volume_integrate!(rows, cols, Svals, root, points, deg)
#         S = SVector(ntuple(d -> sparse(rows[d], cols[d], Svals[d]), Dim))

#         for Iu in CartesianIndices(ntuple(i -> 0:deg, Dim))
#             # Iu[1] is power for x, Iu[2] is power for y, ...
#             if sum([Iu[i] for i = 1:Dim]) <= deg 
#                 # u = (x.^Iu[1]).*(y.^Iu[2]).* ...
#                 u = vec(prod(points.^Tuple(Iu), dims=1))
#                 for Iv in CartesianIndices(ntuple(i -> 0:deg, Dim))
#                     # Iv[1] is power for x, Iv[2] is power for y, ...
#                     if sum([Iv[i] for i = 1:Dim]) <= deg
#                         # v = (x.^Iv[1]).*(y.^Iv[2]).* ...
#                         v = vec(prod(points.^Tuple(Iv), dims=1))
#                         pow_sum = MVector(ntuple(i -> Iu[i] + Iv[i] + 1, Dim))
#                         for d = 1:Dim 
#                             vSu = dot(v, skew_matvec(S[d], u))
#                             pow_sum[d] -= 1 
#                             if pow_sum[d] == 0 
#                                 # in this case, Iu[d] + Iv[d] = 0
#                                 integral = 0.0 
#                             else 
#                                 integral = 0.5*(Iu[d] - Iv[d])/prod(pow_sum)
#                             end 
#                             if abs(integral) < 1e-13
#                                 @test isapprox(vSu, integral, atol=1e-10)
#                             else
#                                 @test isapprox(vSu, integral)
#                             end
#                             pow_sum[d] += 1
#                         end
#                     end
#                 end
#             end
#         end
#         # clear memory from rows, cols, Svals 
#         for d = 1:Dim
#             resize!(rows[d], 0)
#             resize!(cols[d], 0)
#             resize!(Svals[d], 0) 
#         end
#     end
# end 

# @testset "test cut_volume_integrate!: dimension $Dim" for Dim in 2:3

#     # use a unit HyperRectangle 
#     root = Cell(SVector(ntuple(i -> 0.0, Dim)),
#                 SVector(ntuple(i -> i == 1 ? 2.0 : 1.0, Dim)),
#                 CellData(Vector{Int}(), Vector{Int}()))

#     # define a level-set that cuts the HyperRectangle in half 
#     num_basis = 1
#     xc = 0.5*ones(Dim, num_basis)
#     xc[1, 1] = 1.0
#     nrm = zeros(Dim, num_basis)
#     nrm[1, 1] = 1.0
#     tang = zeros(Dim, Dim-1, num_basis)
#     tang[:, :, 1] = nullspace(reshape(nrm[:, 1], 1, Dim))
#     crv = zeros(Dim-1, num_basis)
#     rho = 100.0*num_basis    
#     levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

#     # # define a level-set that cuts the HyperRectangle into a Simplex 
#     # num_basis = Dim
#     # xc = zeros(Dim, num_basis)
#     # nrm = zeros(Dim, num_basis)
#     # tang = zeros(Dim, Dim-1, num_basis)
#     # crv = zeros(Dim-1, num_basis)
#     # for d = 1:Dim 
#     #     idx = d
#     #     xc[d, idx] = 1.0
#     #     nrm[:, idx] = ones(Dim)/sqrt(Dim)
#     #     tang[:, :, idx] = nullspace(reshape(nrm[:,idx], 1, Dim))
#     # end
#     # rho = 100.0*num_basis    
#     # levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

#     # storage for sparse matrix information 
#     rows = Array{Array{Int64}}(undef, Dim)
#     cols = Array{Array{Int64}}(undef, Dim)
#     Svals = Array{Array{Float64}}(undef, Dim)
#     for d = 1:Dim
#         rows[d] = Int[] 
#         cols[d] = Int[] 
#         Svals[d] = Float64[]
#     end

#     # for each polynomial degree `deg` from 1 to max_degree, we form all
#     # possible monomials u and v such that their total degree is `deg` or less.
#     # These monomials are evaluated at the random nodes `points`.  Then we
#     # contract these arrays with the skew symmetric SBP operator in each
#     # direction, and check for accuracy against the expected integral over a
#     # unit HyperRectangle.
#     max_degree = 1
#     for deg = 1:max_degree
#         num_basis = binomial(Dim + deg, Dim)
#         points = randn(Dim, num_basis) .+ 0.5*ones(Dim)
#         root.data.points = 1:num_basis        
#         CutDGD.cut_volume_integrate!(rows, cols, Svals, root, levset, points, deg)
#         S = SVector(ntuple(d -> sparse(rows[d], cols[d], Svals[d]), Dim))

#         for Iu in CartesianIndices(ntuple(i -> 0:deg, Dim))
#             # Iu[1] is power for x, Iu[2] is power for y, ...
#             if sum([Iu[i] for i = 1:Dim]) <= deg 
#                 # u = (x.^Iu[1]).*(y.^Iu[2]).* ...
#                 u = vec(prod(points.^Tuple(Iu), dims=1))
#                 for Iv in CartesianIndices(ntuple(i -> 0:deg, Dim))
#                     # Iv[1] is power for x, Iv[2] is power for y, ...
#                     if sum([Iv[i] for i = 1:Dim]) <= deg
#                         # v = (x.^Iv[1]).*(y.^Iv[2]).* ...
#                         v = vec(prod(points.^Tuple(Iv), dims=1))
#                         pow_sum = MVector(ntuple(i -> Iu[i] + Iv[i] + 1, Dim))
#                         for d = 1:Dim 
#                             vSu = dot(v, skew_matvec(S[d], u))
#                             pow_sum[d] -= 1 
#                             if pow_sum[d] == 0 
#                                 # in this case, Iu[d] + Iv[d] = 0
#                                 integral = 0.0 
#                             else 
#                                 integral = 0.5*(Iu[d] - Iv[d])/prod(pow_sum)
#                             end 
#                             if abs(integral) < 1e-13
#                                 @test isapprox(vSu, integral, atol=1e-10)
#                             else
#                                 @test isapprox(vSu, integral)
#                             end
#                             pow_sum[d] += 1
#                         end
#                     end
#                 end
#             end
#         end
#         # clear memory from rows, cols, Svals 
#         for d = 1:Dim
#             resize!(rows[d], 0)
#             resize!(cols[d], 0)
#             resize!(Svals[d], 0) 
#         end
#     end
# end 