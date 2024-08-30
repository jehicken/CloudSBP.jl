# Tests related to the orthogonal polynomials 

@testset "test proriol_poly in two dimensions" begin
    degree = 4
    x1d, w1d = CloudSBP.lg_nodes(degree+1)
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
            CloudSBP.proriol_poly!(view(P, :, ptr), xq[1,:], xq[2,:], i, j, work)
            ptr += 1
        end
    end
    # verify that integral inner products produce identity
    innerprod = P'*diagm(wq)*P
    @test isapprox(innerprod, Matrix(1.0I, num_basis, num_basis))
    # this should also work with poly_basis!
    CloudSBP.poly_basis!(P, degree, xq, work, Val(2))
    innerprod = P'*diagm(wq)*P
    @test isapprox(innerprod, Matrix(1.0I, num_basis, num_basis))
end

@testset "test proriol_poly in three dimensions" begin
    degree = 4 
    x1d, w1d = CloudSBP.lg_nodes(2*degree)
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
                CloudSBP.proriol_poly!(view(P, :, ptr), xq[1,:], xq[2,:],
                                     xq[3,:], i, j, k, work)
                ptr += 1
            end
        end
    end
    # verify that integral inner products produce identity
    innerprod = P'*diagm(wq)*P
    @test isapprox(innerprod, Matrix(1.0I, num_basis, num_basis))
    # this should also work with poly_basis!
    CloudSBP.poly_basis!(P, degree, xq, work, Val(3))
    innerprod = P'*diagm(wq)*P
    @test isapprox(innerprod, Matrix(1.0I, num_basis, num_basis))
end

@testset "test diff_proriol_poly! in two dimension" begin
    degree = 4
    x1d, w1d = CloudSBP.lg_nodes(degree+1)
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
    CloudSBP.poly_basis_derivatives!(dV, degree, x, Val(2))
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
            CloudSBP.diff_proriol_poly!(dP, view(x,1,:), view(x,2,:), i, j)
            xc[:,:] = x[:,:]
            xc[2,:] .-= complex(0.0, eps_step)
            CloudSBP.proriol_poly!(Pc, view(xc,1,:), view(xc,2,:), i, j, work)
            @test isapprox(-imag(Pc)/eps_step, dP[:,2], atol=1e-13)
            @test isapprox(-imag(Pc)/eps_step, dV[:,ptr,2], atol=1e-13)
            xc[1,:] .-= complex(0.0, eps_step)
            CloudSBP.proriol_poly!(Pc, view(xc,1,:), view(xc,2,:), i, j, work)
            @test isapprox(-imag(Pc)/eps_step - dP[:,2], dP[:,1], atol=1e-13)
            @test isapprox(-imag(Pc)/eps_step - dP[:,2], dV[:,ptr,1],
                           atol=1e-13)
            ptr += 1
        end
    end
end