"""
    quadrature!(xq, wq, rect, x1d, w1d)

Builds a tensor-product quadrature for the domain defined by `rect` using the 
one-dimensional quadratrure defined by the nodes `x1d` and the weights `w1d`.  
The tensor-product quadrature nodes and weights are given by `xq` and `wq`, 
respectively.  These are stored in 2-d and 1-d array formats.
"""
function quadrature!(xq, wq, rect::HyperRectangle{Dim, T}, x1d, w1d
                     ) where {Dim, T}
    @assert( size(xq,1) == Dim && size(xq,2) == size(x1d,1)^Dim && 
             size(wq,1) == size(x1d,1)^Dim,
            "xq and/or wq are not sized correctly" )
    # make a shallow copy of xq with size (Dim, size(x1d,1), size(x1d,1),...)
    xnd = reshape(xq, (Dim, ntuple(i -> size(x1d,1), Dim)...))
    for I in CartesianIndices(xnd)
        # I[1] is the coordinate, so I[I[1] + 1] is the index for that coord
        xnd[I] = (rect.origin[I[1]] + 
                  (x1d[I[I[1]+1]] + 1.0)*0.5*rect.widths[I[1]])
    end
    wnd = reshape(wq, ntuple(i -> size(w1d,1), Dim))
    fill!(wq, one(T))
    for I in CartesianIndices(wnd)
        for d = 1:Dim
            wnd[I] *= w1d[I[d]]*0.5*rect.widths[d]
        end
    end
end

"""
    face_quadrature!(xq, wq, rect, x1d, w1d, dir)

Builds a tensor-product quadrature rule on a `Dim-1` hyperrectangle.  The 
quadrature is returned in the updated arrays `xq` and `wq`, which hold the 
locations and weights, respectively.  The rule is based on the one-dimensional 
quadrature defined by `x1d` and `w1d`.  The `Dim-1` hyperrectuangle has a 
boundary defined by `rect`, which has a unit normal in the Cartesian index 
`dir`.  Note that `rect` is defined in `Dim` dimensions, and has `rect.widths
[dir] = 0`, i.e. it has zero width in the `dir` direction.
"""
function face_quadrature!(xq, wq, rect::HyperRectangle{Dim, T}, x1d, w1d,
                          dir::Int) where {Dim, T}
    @assert( dir >= 1 && dir <= Dim, "invalid dir index" )
    @assert( size(xq,1) == Dim && size(xq,2) == size(x1d,1)^(Dim-1) && 
             size(wq,1) == size(x1d,1)^(Dim-1),
            "xq and/or wq are not sized correctly" )
    # make a shallow copy of xq with size (Dim+1, size(x1d,1), size(x1d,1),...)
    xnd = reshape(xq, (Dim, ntuple(i -> i == dir ? 1 : size(x1d,1), Dim)...))
    for I in CartesianIndices(xnd)
        # I[1] is the coordinate, so I[I[1] + 1] is the index for that coord
        xnd[I] = (rect.origin[I[1]] + 
                  (x1d[I[I[1]+1]] + 1.0)*0.5*rect.widths[I[1]])
    end
    wnd = reshape(wq, ntuple(i -> i == dir ? 1 : size(w1d,1), Dim))
    fill!(wq, one(T))
    tangent_indices = ntuple(i -> i >= dir ? i+1 : i, Dim-1)
    for I in CartesianIndices(wnd)
        for d in tangent_indices
            wnd[I] *= w1d[I[d]]*0.5*rect.widths[d]
        end
    end
end

struct DGDWorkSpace{T, Dim}
    Vc::Vector{T}
    Vq::Vector{T}
    C::Vector{T}
    xc_trans::Matrix{T}
    xq_trans::Matrix{T}
    lower::Vector{T}
    upper::Vector{T}
    dx::Vector{T}
    xavg::Vector{T}
    subwork::Vector{T}

    """
        work = DGDWorkSpace{T,Dim}(degree, max_stencil, max_quad)

    Get a work-space struct to avoid allocations in dgd_basis!.
    """
    function DGDWorkSpace{T, Dim}(degree::Int, max_stencil::Int,
                                  max_quad::Int) where {T,Dim}
        num_basis = binomial(Dim + degree, Dim)
        #num_basis = (degree+1)^Dim
        Vc = zeros(T, (max_stencil*num_basis))
        Vq = zeros(T, (max_quad*num_basis))
        C = zeros(T, (num_basis*max_stencil))
        xc_trans = zeros(T, (Dim, max_stencil))
        xq_trans = zeros(T, (Dim, max_quad))
        lower = zeros(T, (Dim))
        upper = zeros(T, (Dim))
        dx = zeros(T, (Dim))
        xavg = zeros(T, (Dim))
        subwork = zeros(T, ((Dim+1)*max(max_stencil, max_quad)))
        return new(Vc, Vq, C, xc_trans, xq_trans, lower, upper, dx, xavg, 
                   subwork)
    end
end

"""
    dgd_basis!(phi, degree, xc, xq, work, Val(Dim))

Evaluates the DGD basis functions `phi` at the points `xq`.  The basis
functions are of total degree `degree`, and they are defined by the centers 
`xc`.  The array `work` is used for temporary storage, and the value type `Dim` 
is used to dispatch on the relevant dimension.

If `ndims(phi) > 2`, it is assumed that both the basis functions (stored in phi
[:,:,q]) and their derivatives (stored in phi[:,:,2:Dim+1]) are to be 
computed.  Otherwise, the basis functions are computed and returned in the 2d 
array `phi[:,:]`.
"""
function dgd_basis!(phi, degree, xc, xq, work, ::Val{Dim}) where {Dim}
    @assert( size(phi,1) == size(xq,2), "phi and xq have inconsistent sizes")
    @assert( size(xc,1) == size(xq,1) == Dim, "xc/xq/Dim are inconsistent")

    # Get aliases to arrays stored in work
    num_basis = binomial(Dim + degree, Dim)
    #num_basis = (degree+1)^Dim 
    Vc = reshape(view(work.Vc, 1:size(xc,2)*num_basis), (size(xc,2), num_basis))
    C = reshape(view(work.C, 1:size(xc,2)*num_basis), (num_basis, size(xc,2)))
    Vq = reshape(view(work.Vq, 1:size(xq,2)*num_basis), (size(xq,2), num_basis))
    xc_trans = view(work.xc_trans, :, 1:size(xc,2))
    xq_trans = view(work.xq_trans, :, 1:size(xq,2))
    lower = work.lower; upper = work.upper; dx = work.dx; xavg = work.xavg

    # apply an affine transformation to the points xc and xq
    lower[:] = minimum([real.(xc) real.(xq)], dims=2)
    upper[:] = maximum([real.(xc) real.(xq)], dims=2)
    dx[:] = upper - lower 
    xavg[:] = 0.5*(upper + lower)
    dx[:] .*= 1.001
    #xc_trans = zero(xc) 
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end 
    #xq_trans = zero(xq)
    for I in CartesianIndices(xq)
        xq_trans[I] = (xq[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the poly basis and get DGD coefficients (stored in C)
    poly_basis!(Vc, degree, xc_trans, work.subwork, Val(Dim))
    #tensor_basis!(Vc, degree, xc_trans, work.subwork, Val(Dim))
    C[:,:] = pinv(Vc)
    #C = solveBasisCoefficients(Vc, degree)
    fill!(phi, zero(eltype(phi)))
    if ndims(phi) == 2
        # compute just the basis functions, not the derivatives 
        poly_basis!(view(Vq, :, :), degree, xq_trans, work.subwork, Val(Dim))
        #tensor_basis!(view(Vq, :, :), degree, xq_trans, work.subwork, Val(Dim))
        phi[:,:] = Vq*C
        #solve_min_norm!(phi', Vc, Vq')
    else 
        # compute both the basis and their derivatives 
        Vq = zeros(size(xq,2), num_basis, Dim+1)
        poly_basis!(view(Vq, :, :,1), degree, xq_trans, work.subwork, Val(Dim))
        #tensor_basis!(view(Vq, :, :, 1), degree, xq_trans, work.subwork,
        #              Val(Dim))
        poly_basis_derivatives!(view(Vq, :, :, 2:Dim+1), degree, xq_trans,
                                Val(Dim))
        #tensor_basis_derivatives!(view(Vq, :, :, 2:Dim+1), degree, xq_trans,
        #                          Val(Dim))
        phi[:,:,1] = Vq[:,:,1]*C
        #solve_min_norm!(phi[:,:,1]', Vc, Vq[:,:,1]')
        for d = 1:Dim
            #solve_min_norm!(phi[:,:,d+1]', Vc, Vq[:,:,d+1]')/dx[d]
            phi[:,:,d+1] = Vq[:,:,d+1]*C/dx[d]
        end
    end
end

function solveBasisCoefficients(V, degree)
    return pinv(V)
    # num_center = size(V,1)
    # num_poly = size(V,2)
    # A = zeros(num_center^2 + num_poly - 1, num_center*num_poly)
    # A[1:num_center^2,:] = kron(diagm(ones(num_center)), V)
    # A[num_center^2+1:end,:] = kron(ones(num_center)', 
    #         [zeros(num_poly-1,1) diagm(ones(num_poly-1))])
    # b = zeros(num_center^2 + num_poly - 1,1)
    # b[1:num_center^2] = vec(diagm(ones(num_center)))
    # x = pinv(A)*b 
    # return reshape(x, (num_poly, num_center))
end

"""
    dgd_basis_rev!(xc_bar, phi_bar, degree, xc, xq, Val(Dim))

Reverse mode algorithmic differentiation of `dgd_basis!`.  The output `xc_bar` 
is the derivative of the output (defined implicitly by the input `phi_bar`) as 
a function of the coordinates in `xc`.  As in `dgd_basis!`, the basis functions 
are of total degree `degree`, evaluated at points `xq`, and they are defined by 
the centers `xc`.  The value type `Dim` is used to dispatch on the relevant 
dimension.

*Note*: Only differentiates the version of `dgd_basis!` that returns the basis functions, not their derivatives.
"""
function dgd_basis_rev!(xc_bar, phi_bar, degree, xc, xq, ::Val{Dim}) where {Dim}
    @assert( size(phi_bar,1) == size(xq,2), "phi_bar and xq are inconsistent")
    @assert( size(xc_bar,1) == size(xc,1) == size(xq,1) == Dim,
        "xc_bar/xc/xq/Dim have inconsistent sizes")
    # apply an affine transformation to the points xc and xq
    lower = minimum([xc xq], dims=2)
    upper = maximum([xc xq], dims=2)
    dx = upper - lower
    xavg = 0.5*(upper + lower)
    dx .*= 1.001
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end 
    xq_trans = zero(xq)
    for I in CartesianIndices(xq)
        xq_trans[I] = (xq[I] - xavg[I[1]])/dx[I[1]] - 0.5
    end
    # evaluate the poly basis and get DGD coefficients (stored in C)
    num_basis = binomial(Dim + degree, Dim)
    Vc = zeros(size(xc,2), num_basis)
    work = zeros(size(xq,2)*(Dim+1))
    poly_basis!(Vc, degree, xc_trans, work, Val(Dim))
    C = inv(Vc)
    Vq = zeros(size(xq,2), num_basis)
    poly_basis!(view(Vq, :, :), degree, xq_trans, work, Val(Dim))
    #---------------------------------------------------------------------------
    # start reverse sweep
    # phi[:,:] = Vq[:,:]*C
    C_bar = Vq'*phi_bar
    # C = inv(Vc)
    Vc_bar = -C'*C_bar*C'
    # poly_basis!(Vc, degree, xc_trans, Val(Dim))
    dVc = zeros(size(xc,2), num_basis, Dim)  
    poly_basis_derivatives!(dVc, degree, xc_trans, Val(Dim))
    xc_trans_bar = zero(xc)
    for d = 1:Dim 
        for I in CartesianIndices(Vc_bar)
            xc_trans_bar[d,I[1]] += dVc[I,d]*Vc_bar[I]
        end
    end
    for I in CartesianIndices(xc)
        # xc_trans[I] = (xc[I] - xavg[I[1]])/dx[I[1]] - 0.5
        xc_bar[I] += xc_trans_bar[I]/dx[I[1]]
    end
end
