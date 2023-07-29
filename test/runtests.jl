using CutDGD
using Test
using RegionTrees
using StaticArrays: SVector, @SVector
using LinearAlgebra
using Random 

# Following StaticArrays approach of using repeatable "random" tests
Random.seed!(42)


# Need tests for the following files
# * face.jl
# * mass.jl 
# * first_derivative.jl 


@testset "CutDGD.jl" begin
    
    @testset "test quadrature!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:8
        x1d, w1d = CutDGD.lg_nodes(degree+1) # could also use lgl_nodes
        xq = zeros(Dim, length(x1d)^Dim)
        wq = zeros(length(w1d)^Dim)
        cell = Cell(SVector(ntuple(i -> 0.0, Dim)),
                    SVector(ntuple(i -> 1.0, Dim)),
                    CellData(Vector{Int}(), Vector{Face{2,Float64}}()))
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
                    CellData(Vector{Int}(), Vector{Face{2,Float64}}()))
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

    @testset "test orthogonal polynomials" begin 

        @testset "test proriol_poly in two dimensions" begin
            degree = 4
            x1d, w1d = CutDGD.lg_nodes(degree+1)
            # compute Gauss-Legendre quadrature points for a degenerate square
            ptr = 1
            wq = zeros(length(x1d)^2)
            xq = zeros(2, length(wq))
            for j = 1:length(x1d)
                for i = 1:length(x1d)
                    xq[2,ptr] = x1d[j]
                    xq[1,ptr] = 0.5*(1 - xq[2,ptr])*x1d[i] - 0.5*(xq[2,ptr]+1)
                    wq[ptr] = w1d[j]*0.5*(1 - xq[2,ptr])*w1d[i]
                    ptr += 1
                end
            end
            # evaluate Proriol orthogonal polynomials up to degree
            num_basis = binomial(2 + degree, 2)
            P = zeros(length(wq), num_basis)
            work = zeros(3*length(wq))
            ptr = 1
            for r = 0:degree
                for j = 0:r
                    i = r-j                    
                    CutDGD.proriol_poly!(view(P, :, ptr), xq[1,:], xq[2,:], i, 
                                        j, work)
                    ptr += 1
                end
            end
            # verify that integral inner products produce identity
            innerprod = P'*diagm(wq)*P
            @test isapprox(innerprod, Matrix(1.0I, num_basis, num_basis))
            # this should also work with poly_basis!
            CutDGD.poly_basis!(P, degree, xq, work, Val(2))
            innerprod = P'*diagm(wq)*P
            @test isapprox(innerprod, Matrix(1.0I, num_basis, num_basis))
        end

        @testset "test proriol_poly in three dimensions" begin
            degree = 4 
            x1d, w1d = CutDGD.lg_nodes(2*degree)
            # compute Gauss-Legendre quadrature points for a degenerate cube
            ptr = 1
            wq = zeros(length(x1d)^3)
            xq = zeros(3, length(wq))
            for k = 1:length(x1d)
                for j = 1:length(x1d)
                    for i = 1:length(x1d)
                        xq[3,ptr] = x1d[k]
                        xq[2,ptr] = 0.5*(1 + x1d[j])*(1 - xq[3,ptr]) - 1
                        xq[1,ptr] = -1-0.5*(1 + x1d[i])*(xq[2,ptr] + xq[3,ptr])
                        wq[ptr] = -0.25*(1 - xq[3,ptr])*
                            (xq[2,ptr] + xq[3,ptr])*w1d[i]*w1d[j]*w1d[k]
                        ptr += 1
                    end
                end
            end
            # evaluate Proriol orthogonal polynomials up to degree
            num_basis = binomial(3 + degree, 3)
            P = zeros(length(wq), num_basis)
            work = zeros(4*length(wq))
            ptr = 1
            for r = 0:degree
                for k = 0:r
                    for j = 0:r-k
                        i = r-j-k
                        CutDGD.proriol_poly!(view(P, :, ptr), xq[1,:], xq[2,:],
                                            xq[3,:], i, j, k,work)                              
                        ptr += 1
                    end
                end
            end
            # verify that integral inner products produce identity
            innerprod = P'*diagm(wq)*P
            @test isapprox(innerprod, Matrix(1.0I, num_basis, num_basis))
            # this should also work with poly_basis!
            CutDGD.poly_basis!(P, degree, xq, work, Val(3))
            innerprod = P'*diagm(wq)*P
            @test isapprox(innerprod, Matrix(1.0I, num_basis, num_basis))
        end

        @testset "test diff_proriol_poly! in two dimension" begin
            degree = 4
            x1d, w1d = CutDGD.lg_nodes(degree+1)
            # compute Gauss-Legendre quadrature points for a degenerate square
            ptr = 1
            x = zeros(2, length(x1d)^2)
            for j = 1:length(x1d)
                for i = 1:length(x1d)
                    x[2,ptr] = x1d[j]
                    x[1,ptr] = 0.5*(1 - x[2,ptr])*x1d[i] - 0.5*(x[2,ptr]+1)
                    ptr += 1
                end
            end

            dV = zeros(size(x,2), binomial(degree+2,2), 2)
            CutDGD.poly_basis_derivatives!(dV, degree, x, Val(2))
            # test the derivatives using the copmlex-step method 
            eps_step = 1e-60
            num_basis = binomial(degree + 2, 2)
            dP = zeros(size(x,2), 2)
            xc = similar(x, ComplexF64)
            Pc = zeros(ComplexF64, size(x,2))
            work = zeros(ComplexF64, 3*size(x,2))
            ptr = 1
            for r = 0:degree
                for j = 0:r
                    i = r-j
                    CutDGD.diff_proriol_poly!(dP, view(x,1,:), view(x,2,:), i, j)
                    xc[:,:] = x[:,:]
                    xc[2,:] .-= complex(0.0, eps_step)
                    CutDGD.proriol_poly!(Pc, view(xc,1,:), view(xc,2,:), i, j,
                                        work)
                    @test isapprox(-imag(Pc)/eps_step, dP[:,2], atol=1e-13)
                    @test isapprox(-imag(Pc)/eps_step, dV[:,ptr,2], atol=1e-13)
                    xc[1,:] .-= complex(0.0, eps_step)
                    CutDGD.proriol_poly!(Pc, view(xc,1,:), view(xc,2,:), i, j,
                                        work)
                    @test isapprox(-imag(Pc)/eps_step - dP[:,2], dP[:,1], 
                                   atol=1e-13)
                    @test isapprox(-imag(Pc)/eps_step - dP[:,2], dV[:,ptr,1],
                                   atol=1e-13)
                    ptr += 1
                end
            end
        end

    end

    @testset "test dgd_basis!" begin

        @testset "one-dimensional basis: degree $degree" for degree in 0:4
            cell = Cell(SVector(0.), SVector(1.))
            xc = reshape( collect(Float64, 0:degree), (1, degree+1))
            x1d, w1d = CutDGD.lg_nodes(degree+1)
            wq = zeros(length(w1d))
            xq = zeros(1, length(wq))
            CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
            phi = zeros(length(wq), degree+1, 2)
            #work = CutDGD.dgd_basis_work_array(degree, size(xc,2), length(wq), 
            #                                  Val(1))
            work = CutDGD.DGDWorkSpace{Float64,1}(degree, size(xc,2), length(wq))
            CutDGD.dgd_basis!(phi, degree, xc, xq, work, Val(1))
            for b = 1:size(xc,2)
                # y holds the Lagrange polynomials evaluated at xq 
                y = [ prod( ( (x - xc[1,i])/(xc[1,b] - xc[1,i]) 
                            for i = 1:degree+1 if i != b), init=1.0) 
                                for x in xq ]
                for (i, phi_test) in enumerate(y)
                    @test isapprox(phi_test, phi[i, b, 1])
                end 
                # dy holds the derivative of the Lagrange polynomials at xq 
                dy = [ sum( (prod( ((x - xc[1,i])/(xc[1,b] - xc[1,i]) 
                             for i = 1:degree+1 if i != b && i != j), init=1.0)
                                /(xc[1,b] - xc[1,j])
                       for j = 1:degree+1 if j != b), init=0.0) for x in xq]
                for (i, dphi_test) in enumerate(dy)
                    @test isapprox(dphi_test, phi[i, b, 2], atol=1e-12)
                end
            end
        end
        
        @testset "two-dimensional basis" begin
            cell = Cell(SVector(0., 0.), SVector(0.5, 0.5))
            degree = 1
            xc = [0.0 1.0 0.0; 0.0 0.0 1.0]
            x1d, w1d = CutDGD.lg_nodes(degree+1)
            wq = zeros(length(w1d)^2)
            xq = zeros(2, length(wq))
            CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
            phi = zeros(length(wq), 3, 3)
            #work = CutDGD.dgd_basis_work_array(degree, size(xc,2), length(wq), 
            #                                  Val(2))
            work = CutDGD.DGDWorkSpace{Float64,2}(degree, size(xc,2),
                                                 length(wq))        
            CutDGD.dgd_basis!(phi, degree, xc, xq, work, Val(2))
            psi = [(x,y)-> 1-x-y, (x,y)-> x, (x,y)-> y]
            dpsidx = [(x,y)-> -1., (x,y)-> 1., (x,y)-> 0.]
            dpsidy = [(x,y)-> -1., (x,y)-> 0., (x,y)-> 1.]
            for b = 1:size(xc,2)
                # z holds the Lagrange polynomial at xq 
                z = [psi[b](xq[1,i],xq[2,i]) for i = 1:size(xq,2)] 
                for (i, phi_test) in enumerate(z)
                    @test isapprox(phi_test, phi[i, b, 1], atol=1e-12)
                end 
                # dzdx holds the x derivatives at xq 
                dzdx = [dpsidx[b](xq[1,i],xq[2,i]) for i = 1:size(xq,2)]
                for (i, dphi_test) in enumerate(dzdx)
                    @test isapprox(dphi_test, phi[i, b, 2], atol=1e-14)
                end
                # dzdy holds the y derivatives at xq 
                dzdy = [dpsidy[b](xq[1,i],xq[2,i]) for i = 1:size(xq,2)]
                for (i, dphi_test) in enumerate(dzdy)
                    @test isapprox(dphi_test, phi[i, b, 3], atol=1e-14)
                end
            end
            
            degree = 2
            xc = [0.0 1.0 2.0 0.0 1.0 0.0;
                  0.0 0.0 0.0 1.0 1.0 2.0]
            x1d, w1d = CutDGD.lg_nodes(degree+1)
            wq = zeros(length(w1d)^2)
            xq = zeros(2, length(wq))
            CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
            phi = zeros(length(wq), 6, 3)
            #work = CutDGD.dgd_basis_work_array(degree, size(xc,2), length(wq), 
            #                                  Val(2)) 
            work = CutDGD.DGDWorkSpace{Float64,2}(degree, size(xc,2),
                                                 length(wq)) 
            CutDGD.dgd_basis!(phi, degree, xc, xq, work, Val(2))
            psi = [(x,y)-> 1 - 1.5*x - 1.5*y + 0.5*x^2 + 0.5*y^2 + x*y,
                   (x,y)-> 2*x - x^2 - x*y, 
                   (x,y)-> -0.5*x + 0.5*x^2,
                   (x,y)-> 2*y - y^2 - x*y,
                   (x,y)-> x*y,
                   (x,y)-> -0.5*y + 0.5*y^2]
            dpsidx = [(x,y)-> -1.5 + x + y,
                      (x,y)-> 2 - 2*x - y, 
                      (x,y)-> -0.5 + x,
                      (x,y)-> -y,
                      (x,y)->  y,
                      (x,y)-> 0.0]
            dpsidy = [(x,y)-> -1.5 + y + x,
                      (x,y)-> -x, 
                      (x,y)-> 0.0,
                      (x,y)-> 2 - 2*y - x,
                      (x,y)-> x,
                      (x,y)-> -0.5 + y]
            for b = 1:size(xc,2)
                # z holds the Lagrange polynomial at xq 
                z = [psi[b](xq[1,i],xq[2,i]) for i = 1:size(xq,2)] 
                for (i, phi_test) in enumerate(z)
                    @test isapprox(phi_test, phi[i, b, 1], atol=1e-12)
                end 
                # dzdx holds the x derivatives at xq 
                dzdx = [dpsidx[b](xq[1,i],xq[2,i]) for i = 1:size(xq,2)]
                for (i, dphi_test) in enumerate(dzdx)
                    @test isapprox(dphi_test, phi[i, b, 2], atol=1e-14)
                end
                # dzdy holds the y derivatives at xq 
                dzdy = [dpsidy[b](xq[1,i],xq[2,i]) for i = 1:size(xq,2)]
                for (i, dphi_test) in enumerate(dzdy)
                    @test isapprox(dphi_test, phi[i, b, 3], atol=1e-14)
                end
            end
            
        end

        @testset "three-dimensional basis" begin
            cell = Cell(SVector(0., 0., 0.), SVector(0.5, 0.5, 0.5))
            degree = 1
            xc = [0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
            x1d, w1d = CutDGD.lg_nodes(degree+1)
            wq = zeros(length(w1d)^3)
            xq = zeros(3, length(wq))
            CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
            phi = zeros(length(wq), 4, 4)
            #work = CutDGD.dgd_basis_work_array(degree, size(xc,2), length(wq),
            #                                  Val(3)) 
            work = CutDGD.DGDWorkSpace{Float64,3}(degree, size(xc,2),
                                                 length(wq))
            CutDGD.dgd_basis!(phi, degree, xc, xq, work, Val(3))
            psi = [(x,y,z)-> 1-x-y-z, (x,y,z)-> x, (x,y,z)-> y, (x,y,z)-> z]
            dpsidx = [(x,y,z)-> -1., (x,y,z)-> 1., (x,y,z)-> 0., (x,y,z)-> 0.]
            dpsidy = [(x,y,z)-> -1., (x,y,z)-> 0., (x,y,z)-> 1., (x,y,z)-> 0.]
            dpsidz = [(x,y,z)-> -1., (x,y,z)-> 0., (x,y,z)-> 0., (x,y,z)-> 1.]
            for b = 1:size(xc,2)
                # z holds the Lagrange polynomial at xq 
                z = [psi[b](xq[1,i],xq[2,i],xq[3,i]) for i = 1:size(xq,2)] 
                for (i, phi_test) in enumerate(z)
                    @test isapprox(phi_test, phi[i, b, 1], atol=1e-12)
                end 
                # dzdx holds the x derivatives at xq 
                dzdx = [dpsidx[b](xq[1,i],xq[2,i],xq[3,i]) for i = 1:size(xq,2)]
                for (i, dphi_test) in enumerate(dzdx)
                    @test isapprox(dphi_test, phi[i, b, 2], atol=1e-14)
                end
                # dzdy holds the y derivatives at xq 
                dzdy = [dpsidy[b](xq[1,i],xq[2,i],xq[3,i]) for i = 1:size(xq,2)]
                for (i, dphi_test) in enumerate(dzdy)
                    @test isapprox(dphi_test, phi[i, b, 3], atol=1e-14)
                end
                # dzdz holds the z derivatives at xq...sorry, that is confusing
                dzdz = [dpsidz[b](xq[1,i],xq[2,i],xq[3,i]) for i = 1:size(xq,2)]
                for (i, dphi_test) in enumerate(dzdz)                    
                    @test isapprox(dphi_test, phi[i, b, 4], atol=1e-14)
                end
            end
        end

        @testset "test for interpolation: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
            cell = Cell(SVector(ntuple(i -> 0.0, Dim)), 
            SVector(ntuple(i -> 1.0, Dim)))
            num_basis = binomial(degree + Dim, Dim)
            xc = 2*rand(Dim, num_basis) .- 1

            # get quadrature points; we don't need to use the quadrature points 
            # for this particular test, but it doesn't hurt either
            x1d, w1d = CutDGD.lg_nodes(degree+1)
            wq = zeros(length(w1d)^Dim)
            xq = zeros(Dim, length(wq))
            CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)

            # evaluate the DGD representation of the monomials 
            phi = zeros(length(wq), num_basis)
            #work = CutDGD.dgd_basis_work_array(degree, size(xc,2), length(wq),
            #                                  Val(Dim)) 
            work = CutDGD.DGDWorkSpace{Float64,Dim}(degree, size(xc,2),
                                                   length(wq))
            CutDGD.dgd_basis!(phi, degree, xc, xq, work, Val(Dim))
            Vc = zeros(num_basis, num_basis)
            CutDGD.monomial_basis!(Vc, degree, xc, Val(Dim))            
            poly_at_quad = phi*Vc

            # evaluate the monomials at the quadrature points
            Vq = zeros(length(wq), num_basis)
            CutDGD.monomial_basis!(Vq, degree, xq, Val(Dim))

            @test isapprox(poly_at_quad, Vq)
        end

        @testset "test dgd_basis_rev!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
            cell = Cell(SVector(ntuple(i -> 0.0, Dim)), 
                        SVector(ntuple(i -> 0.5, Dim)))
            num_basis = binomial(degree + Dim, Dim)
            xc = rand(Dim, num_basis)
            x1d, w1d = CutDGD.lg_nodes(degree+1)
            wq = zeros(length(w1d)^Dim)
            xq = zeros(Dim, length(wq))
            CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)

            basis_weights = rand(num_basis)
            phi_bar = wq*basis_weights'
            xc_bar = zero(xc)
            CutDGD.dgd_basis_rev!(xc_bar, phi_bar, degree, xc, xq, Val(Dim))

            # the objective is int_{cell} sum_{i=1}^n_k phi(x)*basis_weight dx 
            # we will consider a directional derivative in the direction v 

            # get the analytical derivative 
            v = rand(size(xc))
            dJdv = sum(v.*xc_bar)

            # get the complex-step approximation 
            ceps = 1e-60
            xc_cs = complex.(xc, v*ceps)
            xq_cs = complex.(xq, 0.0)
            phi_cs = zeros(ComplexF64, length(wq), num_basis)
            #work_cs = CutDGD.dgd_basis_work_array(degree, size(xc,2),
            #                                     length(wq), Val(Dim), 
            #                                     type=eltype(xc_cs))
            work_cs = CutDGD.DGDWorkSpace{ComplexF64,Dim}(degree, size(xc,2),
                                                         length(wq))
            CutDGD.dgd_basis!(phi_cs, degree, xc_cs, xq_cs, work_cs, Val(Dim))
            dJdv_cs = imag(wq'*phi_cs*basis_weights)/ceps

            @test isapprox(dJdv, dJdv_cs)
        end

    end # test dgd_basis!

    # @testset "test diagonal mass matrix" begin

    #     @testset "test accuracy of diag_mass: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    #         # get the centers and build the mesh
    #         root = Cell(SVector(ntuple(i -> 0.0, Dim)), 
    #                     SVector(ntuple(i -> 1.0, Dim)),
    #                     CellData(Vector{Int}(), Vector{Face{2,Float64}}()))
    #         num_basis = binomial(degree + Dim, Dim)
    #         xc = rand(Dim, 10*num_basis)
    #         CutDGD.refine_on_points!(root, xc)
    #         for cell in allleaves(root)
    #             split!(cell, CutDGD.get_data)
    #         end
    #         CutDGD.build_nn_stencils!(root, xc, degree)

    #         # get the mass matrix and check accuracy by integrating monomials
    #         mass = CutDGD.diag_mass(root, xc, degree)
    #         for d = 1:Dim
    #             for p = 0:degree 
    #                 integral = vec(xc[d,:].^p)'*mass 
    #                 @test isapprox(integral, 1/(p+1), atol=1e-12) 
    #             end
    #         end
    #     end

    #     @testset "test mass_obj_grad!: dimension $Dim, degree $degree" for Dim in 1:3, degree in 0:4
    #         # get the centers and build the mesh 
    #         root = Cell(SVector(ntuple(i -> 0.0, Dim)), 
    #                     SVector(ntuple(i -> 1.0, Dim)),
    #                     CellData(Vector{Int}(), Vector{Face{2,Float64}}()))
    #         num_basis = binomial(degree + Dim, Dim)
    #         xc = rand(Dim, 10*num_basis)
    #         CutDGD.refine_on_points!(root, xc)
    #         for cell in allleaves(root)
    #             split!(cell, CutDGD.get_data)
    #         end
    #         CutDGD.build_nn_stencils!(root, xc, degree)

    #         # get the gradient; the targets here are irrelevant
    #         mass_targ = rand(size(xc,2))
    #         grad = zeros(length(xc))
    #         CutDGD.mass_obj_grad!(grad, root, xc, degree, mass_targ)
    #         println("norm(grad) = ",norm(grad))

    #         # get the analytical directional derivative 
    #         v = rand(Dim, 10*num_basis)
    #         dJdv = grad'*vec(v)

    #         # check the gradient against the complex-step version 
    #         ceps = 1e-60
    #         xc_cs = complex.(xc, ceps*v)
    #         obj = CutDGD.mass_obj(root, xc_cs, degree, mass_targ)
    #         dJdv_cs = imag(obj)/ceps 

    #         @test isapprox(dJdv, dJdv_cs)
    #     end

    # end

end
