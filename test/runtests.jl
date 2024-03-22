using CutDGD
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using LevelSets
using CxxWrap
using SparseArrays

# Following StaticArrays approach of using repeatable "random" tests
Random.seed!(42)

# Need tests for the following files
# * mass.jl 
# * first_derivative.jl 


@testset "CutDGD.jl" begin
    
    @testset "test first-derivative routines" begin 
        include("test_first_derivative.jl")
    end
    
    if false
    @testset "test utility routines" begin 
        include("test_utils.jl")
    end

    @testset "test quadrature routines" begin 
        include("test_quadrature.jl")
    end

    @testset "test orthogonal polynomials" begin 
        include("test_orthopoly.jl")
    end

    @testset "test face-related routines" begin 
        include("test_face.jl")
    end

    @testset "test mesh routines" begin 
        include("test_mesh.jl")
    end

    @testset "test norm routines" begin
        include("test_norm.jl")
    end
    end

    if false
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
    end 

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
