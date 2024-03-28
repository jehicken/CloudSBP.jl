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
