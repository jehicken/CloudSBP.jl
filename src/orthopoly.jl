"""
    x, w = lgl_nodes(N [, T=Float64])

Computes the Legendre-Gauss-Lobatto (LGL) quadrature nodes `x` and weights `w` 
on the interval [-1,1].  The LGL nodes are the zeros of (1-x^2)*P'_N(x), where 
P_N(x) denotes the Nth Legendre polynomial.

Julia version adapted from Matlab code written by Greg von Winckel - 04/17/2004
Contact: gregvw@chtm.unm.edu
"""
function lgl_nodes(N, T::DataType=Float64)
    N1 = N+1
    # Use the Chebyshev-Gauss-Lobatto nodes as an initial guess
    x = -cos.(π*T[0:N;]/N)
    # The Legendre Vandermonde Matrix 
    P = zeros(T, (N1,N1))
    # Compute P_(N) using the recursion relation; compute its first and second
    # derivatives and update x using the Newton-Raphson method.
    xold = (T)(2)
    while maximum(abs, real(x .- xold)) > eps(real((T)(1)))
        xold = x
        P[:,1] .= one(T)
        P[:,2] = x
        for k=2:N
            P[:,k+1] = ((2k-1)*x.*P[:,k]-(k-1)*P[:,k-1])/k
        end
        x = xold - ( x.*P[:,N1]-P[:,N] )./( N1*P[:,N1] )           
    end
    w = T(2)./(N*N1*P[:,N1].^2)
    return x, w
end

"""
    x, w = lg_nodes(N [, T=Float64])

Computes the Legendre-Gauss (LG) quadrature nodes `x` and weights `w` on the 
interval [-1,1].  The LG nodes are the zeros of P_N(x), where P_N(x) denotes 
the Nth Legendre polynomial.

Julia version adapted from Matlab code written by Greg von Winckel - 02/25/2004
Contact: gregvw@chtm.unm.edu 
"""
function lg_nodes(N, T::DataType=Float64)
    Nm1 = N-1; Np1 = N+1
    N == 1 ? xu = T[0.0] : xu = LinRange{T}(-1, 1, N) 
    # initial guess
    x = -cos.((2*T[0:Nm1;] .+ 1)*pi/(2*Nm1+2)) - (0.27/N)*sin.(pi*xu*Nm1/Np1)
    # Legendre-Gauss Vandermonde Matrix and its derivative
    L = zeros(T, (N,Np1))
    Lp = zeros(T, (N))
    # compute the zeros of the Legendre Polynomial using the recursion relation
    # and Newton's method; loop until new points are uniformly within epsilon of
    # old points
    xold = (T)(2)
    iter = 1; maxiter = 30
    while (maximum(abs, real(x .- xold)) > 0.1*eps(real((T)(1))) && 
           iter < maxiter)
        iter += 1
        L[:,1] .= one(T)
        L[:,2] = x
        for k = 2:N
            L[:,k+1] = ((2*k-1)*x.*L[:,k]-(k-1)*L[:,k-1])/k
        end
        Lp[:] = (Np1)*(L[:,N] .- x.*L[:,Np1])./(1 .- x.^2)
        xold = x
        x -= L[:,Np1]./Lp
    end
    w = T(2)./((1 .- x.^2).*Lp.^2)*(Np1/N)^2
    return x, w
end

"""
    jacobi_poly!(p, x, alpha, beta, N, work)

Evaluate Jacobi polynomial at the points `x` and return in `p`.  Based on 
JacobiP in Hesthaven and Warburton's nodal DG book.  `alpha` and `beta` are 
parameters that define the type of Jacobi Polynomial (alpha + beta != 1).  `N` 
is the polynomial degree, not the number of nodes.  The array `work` should be 
twice the length of `x`.
"""
function jacobi_poly!(p::AbstractVector{T}, x::AbstractVector{T}, alpha, beta, 
                      N::Int, work::AbstractVector{T}) where {T}
    @assert( alpha + beta != -1 )
    @assert( alpha > -1 && beta > -1 )
    @assert( length(p) == length(x) )
    @assert( length(work) >= 2*length(x) )
    # Initial values P_0(x) and P_1(x)
    gamma0 = ((2^(alpha+beta+1))/(alpha+beta+1))*gamma(alpha+1)*gamma(beta+1)/
    gamma(alpha+beta+1)
    p[:] = ones(size(x))/sqrt(gamma0)
    if N == 0
        return nothing
    end
    P_0 = view(work, 1:length(x))
    P_0[:] = p
    gamma1 = (alpha+1)*(beta+1)*gamma0/(alpha+beta+3)
    p[:] = 0.5*((alpha+beta+2).*x .+ (alpha-beta))/sqrt(gamma1)
    if N == 1
        return nothing
    end
    # Henceforth, P_0 denotes P_{i} and P_1 denotes P_{i+1}
    save = view(work, length(x)+1:2*length(x))
    P_1 = view(p, :)
    # repeat value in recurrence
    aold = (T(2)./(2+alpha+beta))*sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
    for i = 1:N-1
        h1 = 2*i + alpha + beta
        anew = (T(2)./(h1+2))*sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/
        ((h1+1)*(h1+3)))
        bnew = -(alpha^2 - beta^2)/(h1*(h1+2))
        save[:] = P_1
        P_1[:] = (1/anew).*(-aold.*P_0 + (x .- bnew).*P_1)
        P_0[:] = save
        aold = anew
    end
    return nothing
end

"""
    dp = diff_jacobi_poly(x, alpha, beta, N)

Evaluate the first derivative of a Jacobi Polynomial at the points `x`.  See 
the companion code for `jacobipoly` for an explanation of the other parameters.
"""
function diff_jacobi_poly(x::AbstractVector{T}, alpha, beta, N::Int
                          ) where {T}
    @assert( alpha + beta != -1 )
    @assert( alpha > -1 && beta > -1)
    DP_0 = zero(x)
    if N == 0
        size(DP_0,1) > size(DP_0,2) ? (return DP_0) : (return transpose(DP_0))
    end
    gamma0 = ((2^(alpha+beta+1))/(alpha+beta+1))*gamma(alpha+1)*gamma(beta+1)/
    gamma(alpha+beta+1)
    gamma1 = (alpha+1)*(beta+1)*gamma0/(alpha+beta+3)
    DP_1 = ones(size(x)).*0.5*(alpha+beta+2)/sqrt(gamma1)
    if N == 1
        size(DP_1,1) > size(DP_1,2) ? (return DP_1) : (return transpose(DP_1))
    end
    # initialize values P_0(x) and P_1(x) for recurrence
    P_0 = ones(size(x))./sqrt(gamma0)
    P_1 = 0.5*((alpha+beta+2).*x .+ (alpha-beta))/sqrt(gamma1)
    # repeat value in recurrence
    aold = (T(2)./(2+alpha+beta))*sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
    save = zero(x)
    for i = 1:N-1
        h1 = 2*i + alpha + beta
        anew = (T(2)./(h1+2))*sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/
        ((h1+1)*(h1+3)))
        bnew = -(alpha^2 - beta^2)/(h1*(h1+2))
        save = DP_1
        DP_1 = (1/anew).*(-aold.*DP_0 .+ P_1 .+ (x .- bnew).*DP_1)
        DP_0 = save
        save = P_1
        P_1 = (1/anew).*(-aold.*P_0 .+ (x .- bnew).*P_1)
        P_0 = save
        aold = anew
    end
    size(DP_1,1) > size(DP_1,2) ? (return DP_1) : (return transpose(DP_1))
end

"""
    proriol_poly!(p, x, y, i, j, work)

Evaluate a Proriol orthogonal polynomial basis function on the right triangle 
and return in array `p`.  The arrays `x` and `y` give the locations at which to 
evaluate the polynomial, and the integers `i` and `j` define the basis function 
to evaluate; see Hesthaven and Warburton's Nodal DG book, for example, for a 
reference.  `work` is a vector that should be 3 times the length of p.
"""
function proriol_poly!(p::AbstractVector{T}, x::AbstractVector{T}, 
                       y::AbstractVector{T}, i::Int, j::Int, 
                       work::AbstractVector{T}) where {T}
    @assert( i >= 0 && j >= 0 )
    @assert( length(work) >= 3*length(p) )
    xi = view(work, 1:length(x))
    for k = 1:length(x)
        real(y[k]) != 1 ? xi[k] = 2*(1 + x[k])/(1 - y[k]) -1 : xi[k] = -1
    end
    rzero = real(zero(T))
    jacobi_poly!(p, xi, rzero, rzero, i, view(work, length(x)+1:3*length(x)))
    jacobi_poly!(xi, y, 2*i+1, rzero, j, view(work, length(x)+1:3*length(x)))
    p[:] .*= sqrt(2) .* xi .* ((1 .- y).^(i))
    return nothing 
    #return sqrt(2).*jacobi_poly(xi, rzero, rzero, i).*
    #    jacobi_poly(y, 2*i+1, rzero, j) .* ((1 .- y).^(i))
end

"""
    diff_proriol_poly!(dP, x, y, i, j)

Evaluate the derivatives of a Proriol orthogonal polynomial basis function on
the right triangle.  The arrays `x` and `y` give the locations at which to
evaluate the polynomial, and the integers `i` and `j` define the basis function 
to evaluate.  The result is stored in `dP`.
"""
function diff_proriol_poly!(dP::AbstractArray{T,2}, x::AbstractVector{T}, 
                            y::AbstractVector{T}, i::Int, j::Int) where {T}
    xi = zero(x)
    for k = 1:length(x)
        real(y[k]) != 1.0 ? xi[k] = 2*(1 + x[k])./(1 - y[k]) -1 : xi[k] = -1
    end
    fill!(dP, zero(T))
    if i == 0 && j == 0
        return nothing
    end
    # compute some terms for reuse 
    work = zeros(2*length(x))
    Jxi = zero(x)
    jacobi_poly!(Jxi, xi, zero(T), zero(T), i, work)
    Jy = zero(y)
    jacobi_poly!(Jy, y, convert(T, 2*i+1), zero(T), j, work)
    dJxidxi = diff_jacobi_poly(xi, zero(T), zero(T), i)
    dJydy = diff_jacobi_poly(y, convert(T, 2*i+1), zero(T), j)
    if (i > 0)
        # dPdx is only nonzero if i > 0;
        dP[:,1] += sqrt(2).*dJxidxi.*Jy.*2.0.*((1 .- y).^(i-1))
        dP[:,2] = -Jy.*i.*((1 .- y).^(i-1))
    end
    # dPdx is now finished, but dPdy needs additional terms
    dP[:,2] += dJydy.*((1 .- y).^(i))
    dP[:,2] .*= Jxi
    if (i >= 1)
        for k = 1:length(x)
            real(y[k]) != 1.0 ? 
                dP[k,2] += dJxidxi[k]*Jy[k]*2*(1+x[k])*((1-y[k])^(i-2)) : 
                dP[k,2] += 0.0
        end
    end
    dP[:,2] *= sqrt(2)
    return nothing
end

"""
    proriol_poly!(p, x, y, z, i, j, k, work)

Evaluate a Proriol orthogonal polynomial basis function on the right 
tetrahedron and return in array `p`.  The arrays `x`,`y`, and `z` are the 
locations at which to evaluate the polynomial, and the integers `i`,`j`, and 
`k` define the basis function to evaluate; see Hesthaven and Warburton's Nodal 
DG book, for example, for a reference.  The `work` array should be at least 4 
times the length of `p`.
"""
function proriol_poly!(p::AbstractVector{T}, x::AbstractVector{T},
                       y::AbstractVector{T}, z::AbstractVector{T}, i::Int, 
                       j::Int, k::Int, work::AbstractVector{T}) where {T}
    @assert( i >= 0 && j >= 0 && k >= 0 )
    @assert( length(work) >= 2*length(p) )
    xi = view(work, 1:length(p))
    for m = 1:length(x)
        real(y[m]+z[m]) != 0 ? xi[m] = -2*(1+x[m])/(y[m]+z[m]) - 1 : xi[m] = -1
    end 
    rzero = real(zero(T))
    work_view = view(work, length(p)+1:3*length(p))
    jacobi_poly!(p, xi, rzero, rzero, i, work_view)
    eta = view(work, 1:length(p)) # no longer need xi, but just rename alias
    for m = 1:length(x)
        real(z[m]) != 1 ? eta[m] = 2*(1+y[m])/(1-z[m]) - 1 : eta[m] = -1
    end 
    jp = view(work, 3*length(p)+1:4*length(p))
    jacobi_poly!(jp, eta, 2*i+1, rzero, j, work_view)
    p[:] .*= sqrt(8) .* jp .* ((1 .- eta).^(i))
    jacobi_poly!(jp, z, 2*i+2*j+2, rzero, k, work_view)
    p[:] .*= jp .* ((1 .- z).^(i+j))
    return nothing 
    #return sqrt(8).*jacobi_poly(xi, rzero, rzero, i).*
    #    jacobi_poly(eta, 2*i+1, rzero, j).*((1 .- eta).^(i)).*
    #    jacobi_poly(z, 2*i+2*j+2, rzero, k).*((1 .- z).^(i+j))    
end

"""
    diff_proriol_poly!(dP, x, y, z, i, j, k)

Evaluate the derivatives of a Proriol orthogonal polynomial basis function on
the right tetrahedron.  The arrays `x`,`y`, and `z` are the locations at which 
to evaluate the polynomial, and the integers `i`,`j`, and `k` define the basis 
function to evaluate.  The result is stored in `dP`.

*Notes*: the derivatives are computed using the complex-step method (since there
 are many outputs and only 3 inputs); therefore, a different method should be
 used for verification of this method.
"""
function diff_proriol_poly!(dP::AbstractArray{T,2}, x::AbstractVector{T}, 
                            y::AbstractVector{T}, z::AbstractVector{T}, i::Int, 
                            j::Int, k::Int) where {T}
    @assert( size(dP,1) == length(x) == length(y) == length(z) )
    @assert( size(dP,2) == 3 )
    # each node is independent, so use complex step once for each coordinate.
    # Care is needed at the one vertex, where the xi and eta mappings become 
    # singular. To avoid problems, directional derivatives are used.
    eps_step = get_complex_step(T)
    xc = complex.(x, 0)
    yc = complex.(y, 0)
    zc = complex.(z, 0)    
    # compute derivative with respect to z
    zc .-= eps_step*im
    Pc = zero(xc) 
    work = zeros(eltype(xc), 4*length(dP) )
    proriol_poly!(Pc, xc, yc, zc, i, j, k, work)
    dP[:,3] = -imag(Pc)./eps_step
    # compute dPdy = -(Grad P) dot (0,-1,-1) - dPdz
    yc .-= eps_step*im
    proriol_poly!(Pc, xc, yc, zc, i, j, k, work)
    dP[:,2] = -dP[:,3] - imag(Pc)./eps_step
    # compute dPdx = -(Grad P) dot (-1,-1,-1) - dPdz - dPdy
    xc .-= eps_step*im
    proriol_poly!(Pc, xc, yc, zc, i, j, k, work)
    dP[:,1] = -dP[:,3] - dP[:,2] - imag(Pc)./eps_step
    return nothing
end

"""
    poly_basis!(V, degree, x, work, Val(N))

This method and its variants compute the approporiate polynomial basis of total 
degree `degree` at the points `x`.  The type parameter `N` is used to select 
the appropriate spatial dimension.  The size of `work` should be `N+1` times 
the number of rows in `V`.
"""
function poly_basis!(V::AbstractArray{T,2}, degree::Int, x::AbstractArray{T,2}, 
                     work::AbstractVector{T}, ::Val{1}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == degree+1 )
    @assert( length(work) >= 2*size(V,1) )
    for i = 0:degree 
        #V[:,i+1] = jacobi_poly(view(x,1,:), 0.0, 0.0, i)
        jacobi_poly!(view(V,:,i+1), view(x,1,:), 0.0, 0.0, i, work)
    end
    return nothing
end

function poly_basis!(V::AbstractArray{T,2}, degree::Int, x::AbstractArray{T,2}, 
                     work::AbstractVector{T}, ::Val{2}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == binomial(2 + degree, 2) )
    @assert( length(work) >= 3*size(V,1) )
    ptr = 1
    for r = 0:degree
        for j = 0:r 
            i = r - j
            #V[:,ptr] = proriol_poly(view(x,1,:), view(x,2,:), i, j)
            proriol_poly!(view(V, :, ptr), view(x,1,:), view(x,2,:), i, j, work)
            ptr += 1
        end
    end
    return nothing
end

function poly_basis!(V::AbstractArray{T,2}, degree::Int, x::AbstractArray{T,2},
                     work::AbstractVector{T}, ::Val{3}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == binomial(3 + degree, 3) )
    @assert( length(work) >= 4*size(V,1) )
    ptr = 1
    for r = 0:degree
        for k = 0:r
            for j = 0:r-k
                i = r - j - k
                #V[:,ptr] = proriol_poly(view(x,1,:), view(x,2,:), view(x,3,:),
                #                        i, j, k)
                proriol_poly!(view(V, :, ptr), view(x,1,:), view(x,2,:),
                              view(x,3,:), i, j, k, work)
                ptr += 1
            end 
        end
    end
    return nothing
end

"""
    poly_basis_derivatives!(dV, degree, x, Val(N))

This method and its variants compute the first derivatives of polynomial basis 
of total degree `degree` at the points `x`.  The type parameter `N` is used to 
select the appropriate spatial dimension.
"""
function poly_basis_derivatives!(dV::AbstractArray{T,3}, degree::Int, 
                                 x::AbstractArray{T,2}, ::Val{1}) where {T}
    @assert( size(dV,1) == size(x,2) && size(dV,2) == degree+1 && 
             size(dV,3) == 1)
    for i = 0:degree 
        dV[:,i+1] = diff_jacobi_poly(view(x,1,:), 0.0, 0.0, i)
    end
end

function poly_basis_derivatives!(dV::AbstractArray{T,3}, degree::Int, 
                                 x::AbstractArray{T,2}, ::Val{2}) where {T}
    @assert( size(dV,1) == size(x,2) && size(dV,2) == binomial(2 + degree, 2) &&
             size(dV,3) == 2)
    ptr = 1
    for r = 0:degree
        for j = 0:r 
            i = r - j
            diff_proriol_poly!(view(dV,:,ptr,:), view(x,1,:), view(x,2,:), i, j)
            ptr += 1
        end
    end
end

function poly_basis_derivatives!(dV::AbstractArray{T,3}, degree::Int, 
                                 x::AbstractArray{T,2}, ::Val{3}) where {T}
    @assert( size(dV,1) == size(x,2) && size(dV,2) == binomial(3 + degree, 3) &&
             size(dV,3) == 3)
    ptr = 1
    for r = 0:degree
        for k = 0:r
            for j = 0:r-k
                i = r - j - k
                diff_proriol_poly!(view(dV,:,ptr,:), view(x,1,:), view(x,2,:),
                                   view(x,3,:), i, j, k)
                ptr += 1
            end 
        end
    end
    return nothing
end

"""
    monomial_basis!(V, degree, x, Val(N))

This method computes the monomial basis of total degree `degree` at the points 
`x`.  The type parameter `N` is used to select the appropriate spatial 
dimension.  These methods are useful for testing, but given the 
ill-conditioning of the corresponding Vandermonde matrix, they should not be 
used to construct the DGD basis functions.
"""
function monomial_basis!(V::AbstractArray{T,2}, degree::Int,
                         x::AbstractArray{T,2}, ::Val{1}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == degree+1 )
    for i = 0:degree 
        V[:,i+1] = x[1,:].^i
    end
end

function monomial_basis!(V::AbstractArray{T,2}, degree::Int,
                         x::AbstractArray{T,2}, ::Val{2}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == binomial(2 + degree, 2) )
    ptr = 1
    for r = 0:degree
        for j = 0:r 
            i = r - j
            V[:,ptr] = (x[1,:].^i).*(x[2,:].^j)
            ptr += 1
        end
    end
end

function monomial_basis!(V::AbstractArray{T,2}, degree::Int,
                         x::AbstractArray{T,2}, ::Val{3}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == binomial(3 + degree, 3) )
    ptr = 1
    for r = 0:degree
        for k = 0:r
            for j = 0:r-k
                i = r - j - k
                V[:,ptr] = (x[1,:].^i).*(x[2,:].^j).*(x[3,:].^k)
                ptr += 1
            end 
        end
    end
end

"""
    monomial_basis_derivatives!(dV, degree, x, Val(N))

This method and its variants compute the first derivatives of monomial basis 
of total degree `degree` at the points `x`.  The type parameter `N` is used to 
select the appropriate spatial dimension.
"""
function monomial_basis_derivatives!(dV::AbstractArray{T,3}, degree::Int, 
                                     x::AbstractArray{T,2}, ::Val{1}) where {T}
    @assert( size(dV,1) == size(x,2) && size(dV,2) == degree+1 && 
             size(dV,3) == 1)
    for i = 0:degree 
        dV[:,i+1,1] = i*x[1,:].^max(0,i-1)
    end
    return nothing
end

function monomial_basis_derivatives!(dV::AbstractArray{T,3}, degree::Int, 
                                     x::AbstractArray{T,2}, ::Val{2}) where {T}
    @assert( size(dV,1) == size(x,2) && size(dV,2) == binomial(2 + degree, 2) &&
             size(dV,3) == 2)
    ptr = 1
    for r = 0:degree
        for j = 0:r 
            i = r - j
            # V[:,ptr] = (x[1,:].^i).*(x[2,:].^j)
            dV[:,ptr,1] = i*(x[1,:].^max(i-1,0)).*(x[2,:].^j)
            dV[:,ptr,2] = j*(x[1,:].^i).*(x[2,:].^max(j-1,0))
            ptr += 1
        end
    end
    return nothing
end

function monomial_basis_derivatives!(dV::AbstractArray{T,3}, degree::Int, 
                                     x::AbstractArray{T,2}, ::Val{3}) where {T}
    @assert( size(dV,1) == size(x,2) && size(dV,2) == binomial(3 + degree, 3) &&
             size(dV,3) == 3)
    ptr = 1
    for r = 0:degree
        for k = 0:r
            for j = 0:r-k
                i = r - j - k
                # V[:,ptr] = (x[1,:].^i).*(x[2,:].^j).*(x[3,:].^k)
                dV[:,ptr,1] = i*(x[1,:].^max(i-1,0)).*(x[2,:].^j).*(x[3,:].^k)
                dV[:,ptr,2] = j*(x[1,:].^i).*(x[2,:].^max(j-1,0)).*(x[3,:].^k)
                dV[:,ptr,3] = k*(x[1,:].^i).*(x[2,:].^j).*(x[3,:].^max(k-1,0))
                ptr += 1
            end 
        end
    end
    return nothing
end

"""
    tensor_basis!(V, degree, x, Val(N))

This method computes the tensor-product basis of degree `degree` at the points 
`x`.  The type parameter `N` is used to select the appropriate spatial 
dimension.
"""
function tensor_basis!(V::AbstractArray{T,2}, degree::Int,
                       x::AbstractArray{T,2}, work::AbstractVector{T},
                       ::Val{1}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == degree+1 )
    @assert( length(work) >= 2*size(V,1) )
    for i = 0:degree 
        #V[:,i+1] = jacobi_poly(view(x,1,:), 0.0, 0.0, i)
        jacobi_poly!(view(V,:,i+1), view(x,1,:), 0.0, 0.0, i, work)
    end
    return nothing
end

function tensor_basis!(V::AbstractArray{T,2}, degree::Int,
                       x::AbstractArray{T,2}, work::AbstractVector{T},
                       ::Val{2}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == (degree+1)^2 )
    @assert( length(work) >= 3*size(V,1) )
    Jy = view(work, 2*size(V,1)+1:3*size(V,1))
    ptr = 1
    for j = 0:degree 
        jacobi_poly!(Jy, view(x,2,:), 0.0, 0.0, j, work)
        for i = 0:degree
            jacobi_poly!(view(V, :, ptr), view(x,1,:), 0.0, 0.0, i, work) 
            V[:,ptr] .*= Jy[:]
            ptr += 1
        end
    end 
    return nothing
end

function tensor_basis!(V::AbstractArray{T,2}, degree::Int,
                       x::AbstractArray{T,2}, work::AbstractVector{T},
                       ::Val{3}) where {T}
    @assert( size(V,1) == size(x,2) && size(V,2) == (degree+1)^3 )
    @assert( length(work) >= 4*size(V,1) )
    Jy = view(work, 2*size(V,1)+1:3*size(V,1))
    Jz = view(work, 3*view(V,1)+1:4*size(V,1))
    ptr = 1
    for k = 0:degree 
        jacobi_poly!(Jz, view(x,3,:), 0.0, 0.0, k, work)
        for j = 0:degree 
            jacobi_poly!(Jy, view(x,2,:), 0.0, 0.0, j, work)
            for i = 0:degree 
                jacobi_poly!(view(V, :, ptr), view(x,1,:), 0.0, 0.0, i, work) 
                V[:,ptr] .*= Jy[:].*Jz[:]
                ptr += 1
            end
        end
    end
    return nothing
end

function tensor_basis_derivatives!(dV::AbstractArray{T,3}, degree::Int, 
                                   x::AbstractArray{T,2}, ::Val{1}) where {T}
    @assert( size(dV,1) == size(x,2) && size(dV,2) == degree+1 && 
             size(dV,3) == 1)
    for i = 0:degree 
        dV[:,i+1,1] = diff_jacobi_poly(view(x,1,:), 0.0, 0.0, i)
    end
end

function tensor_basis_derivatives!(dV::AbstractArray{T,3}, degree::Int, 
                                   x::AbstractArray{T,2}, ::Val{2}) where {T}
    @assert( size(dV,1) == size(x,2) && size(dV,2) == (degree+1)^2 &&
             size(dV,3) == 2)
    work = zeros(2*size(dV,1))
    Jx = zeros(size(dV,1))
    dJx = zero(Jx)
    Jy = zeros(size(dV,1))
    dJy = zero(Jy)
    ptr = 1
    for j = 0:degree 
        jacobi_poly!(Jy, view(x,2,:), 0.0, 0.0, j, work)
        dJy = diff_jacobi_poly(view(x,2,:), 0.0, 0.0, j)
        for i = 0:degree
            jacobi_poly!(Jx, view(x,1,:), 0.0, 0.0, i, work) 
            dJx = diff_jacobi_poly(view(x,1,:), 0.0, 0.0, i)
            dV[:,ptr,1] = dJx.*Jy 
            dV[:,ptr,2] = Jx.*dJy
            ptr += 1
        end
    end 
    return nothing
end