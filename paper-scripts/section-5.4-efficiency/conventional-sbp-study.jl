module Conventional 
# Solve the annulus problem using conventional finite-difference SBP 

using SummationByPartsOperators
using LinearAlgebra
using SparseArrays

# study parameters
Dim = 2
vel = ones(Dim)
deg = [1;2;3;4]
num1d = [11; 21; 41; 81; 161]

# all boundaries have characteristic BCs applied to them
bc_map = Dict{Any,String}("ib" => "upwind")
for di = 1:Dim 
    bc_map[di*2-1] = "upwind"
    bc_map[di*2] = "upwind"
end

uexact(x::AbstractVector) = exp(sum(x))
dudx(x::AbstractVector) = Dim*exp(sum(x))
uexact(x::AbstractMatrix) = exp.(sum(x,dims=1))'
dudx(x::AbstractMatrix) = Dim*exp.(sum(x,dims=1))'

function levset(x)
    r = norm(x)
    return -(r^2 - 1)*(r^2 - 0.25)
end
function levset_grad!(g, x)
    r = norm(x)
    # dphidr is divided by r to avoid issues
    dphidr = -2*(r^2 - 0.25) - (r^2 - 1)*2
    g[:] = dphidr*x
    return nothing
end

"""
    x = get_mesh(nr, ntheta)

Returns the nodes of a tensor-product mesh that conforms to the annulus defined 
by the above level-set.  There are `nr` nodes in the radial direction and 
`ntheta` nodes in the angular direction (the nodes in the theta direction do 
not overlap at theta = 0).
"""
function get_mesh(nr, ntheta)
    @assert(Dim == 2)
    radius(xi) = (exp(growth*xi) - 1.0)/(exp(growth) - 1.0)
    dradius(xi) = growth*exp(growth*xi)/(exp(growth)-1.0)
    rin = 0.5
    rout = 1.0
    x = zeros(Dim, nr, ntheta)
    for i in axes(x,2) # r loop
        r = (nr-i)*rin/(nr-1) + (i-1)*rout/(nr-1)        
        for j in axes(x,3) # theta loop
            theta = (j-1)*2*pi/ntheta
            x[1,i,j] = r*cos(theta)
            x[2,i,j] = r*sin(theta)
        end
    end
    return x
end

"""
    A, b = build_system(Dxi, Deta, x, dxidx, jac, Mxi, Meta)

Returns the linear system corresponding to linear advection on the annulus.  
The SBP operators `Dxi` and `Deta` are for numerical differentiation in the 
radial and angular directions, respectively.  The coordinates are stored in the 
3D array `x`, the (scaled) mapping Jacobian terms are in the 4D array `dxidx`, 
and the determinant is in `jacH`.  `Mxi` and `Meta` are the 1D mass matrices 
corresponding to `Dxi` and `Deta`.  The system matrix is returned as a sparse 
CSC type.
"""
function build_system(Dxi, Deta, x, dxidx, jac, Mxi, Meta)

    # set up arrays to store sparse matrix information
    num_nodes = size(Dxi,1)*size(Deta,1)
    row = Int[]
    col = Int[]
    vals = Float64[]
    b = zeros(num_nodes)

    function get_index(i, j)
        return (j-1)*size(Dxi,1) + i 
    end

    for j in axes(x,3)
        for i in axes(x,2)
            # contributions due to derivative in the xi direction 
            Dxicol = i
            for ptr in nzrange(Dxi, Dxicol)
                Dxirow = rowvals(Dxi)[ptr]
                Dxival = nonzeros(Dxi)[ptr]
                # Dxi[Dxirow,Dxicol] --> Dxi[(nbr,j),(i,j)]
                append!(row, get_index(Dxirow, j))
                append!(col, get_index(Dxicol, j))
                append!(vals, Dxival*(dxidx[1,1,i,j]*vel[1] + 
                                      dxidx[1,2,i,j]*vel[2]))
            end

            # contributions due to derivative in the eta direction 
            Detacol = j 
            for ptr in nzrange(Deta, Detacol)
                Detarow = rowvals(Deta)[ptr]
                Detaval = nonzeros(Deta)[ptr]
                # Deta[Detarow,Detacol] --> Deta[(i,nbr),(i,j)]
                append!(row, get_index(i, Detarow))
                append!(col, get_index(i, Detacol))
                append!(vals, Detaval*(dxidx[2,1,i,j]*vel[1] +
                                       dxidx[2,2,i,j]*vel[2]))
            end

            # add source terms 
            b[get_index(i,j)] += jac[i,j] * dudx(x[:,i,j])
        end
    end

    # periodic in eta, so only need SATs along xi_min and xi_max 
    for j in axes(x,3)
        i = 1
        # contributions at inner radius 
        velnrm = dxidx[1,1,i,j]*vel[1] + dxidx[1,2,i,j]*vel[2]
        velnrm *= -1.0 # need this because dxidx is not outward pointing at i=1
        if velnrm < 0.0 
            # this is an inflow, use a SAT 
            append!(row, get_index(i, j))
            append!(col, get_index(i, j))
            append!(vals, velnrm/Mxi[i,i])
            b[get_index(i, j)] += velnrm*uexact(x[:,i,j])/Mxi[i,i]
        end
        i = size(Dxi,1)
        # contributions at outer radius 
        velnrm = dxidx[1,1,i,j]*vel[1] + dxidx[1,2,i,j]*vel[2]
        if velnrm < 0.0 
            # this is an inflow, use a SAT 
            append!(row, get_index(i, j))
            append!(col, get_index(i, j))
            append!(vals, -velnrm/Mxi[end,end])
            b[get_index(i, j)] -= velnrm*uexact(x[:,i,j])/Mxi[end,end]
        end
    end

    A = sparse(row, col, vals, num_nodes, num_nodes)
    return A, b
end

L2err = zeros(length(num1d), length(deg))
nodes = zero(L2err)
dx = zero(L2err)
maxerr = zero(L2err)
Annz = zero(L2err)

for (i, nr) in enumerate(num1d)
    ntheta = 6*(nr-1) + 1
    # generate the mesh for given nx 
    x = get_mesh(nr, ntheta)
    num_nodes = size(x,2)*size(x,3)
    dx[i, :] .= 1/nr
    nodes[i, :] .= num_nodes 

    for (dindex, degree) in enumerate(deg)
        println()
        println(repeat("=",80))

        if nr < 9 && degree == 2
            L2err[i,dindex] = NaN
            maxerr[i,dindex] = NaN
            Annz[i,dindex] = NaN
            continue 
        end
        if nr < 13 && degree == 3
            L2err[i,dindex] = NaN
            maxerr[i,dindex] = NaN
            Annz[i,dindex] = NaN
            continue
        end
        if nr < 17 && degree == 4
            L2err[i,dindex] = NaN
            maxerr[i,dindex] = NaN
            Annz[i,dindex] = NaN
            continue
        end

        # generate the SBP operators for degree 
        Deta = periodic_derivative_operator(derivative_order = 1, 
                                            accuracy_order = 2*degree, 
                                            xmin = 0.0, xmax = 2*pi, N = ntheta)
        Meta = mass_matrix(Deta)
        Dxi = derivative_operator(MattssonNordström2004(),
                                  derivative_order = 1,
                                  accuracy_order = 2*degree, #degree+1,
                                  xmin = 0.0, xmax = 1.0, N = nr)
        Mxi = mass_matrix(Dxi)

        # compute the mapping Jacobian and determinant 
        dxdxi = zeros(2,2,nr,ntheta)
        for j in axes(x, 3)
            # dx/dxi 
            mul!(view(dxdxi, 1, 1, :, j), Dxi, view(x, 1, :, j))
            # dy/dxi 
            mul!(view(dxdxi, 2, 1, :, j), Dxi, view(x, 2, :, j))
        end
        for i in axes(x, 2)
            # dx/deta 
            mul!(view(dxdxi, 1, 2, i, :), Deta, view(x, 1, i, :))
            # dy/deta 
            mul!(view(dxdxi, 2, 2, i, :), Deta, view(x, 2, i, :))
        end
        dxidx = zero(dxdxi)
        dxidx[1,1,:,:] =  dxdxi[2,2,:,:]
        dxidx[1,2,:,:] = -dxdxi[1,2,:,:]
        dxidx[2,1,:,:] = -dxdxi[2,1,:,:]
        dxidx[2,2,:,:] =  dxdxi[1,1,:,:]
        jac = zeros(nr, ntheta)        
        jac .= dxdxi[1,1,:,:].*dxdxi[2,2,:,:] - dxdxi[1,2,:,:].*dxdxi[2,1,:,:]

        M = zeros(nr, ntheta)
        for j in axes(x, 3)
            for i in axes(x, 2)
                M[i,j] = Mxi[i,i]*Meta[j,j]
            end
        end
        println("volume error = ",abs(sum(jac.*M) - pi*(1 - 0.25)))

        A, b = build_system(sparse(Dxi), sparse(Deta), x, dxidx, jac, Mxi, 
                            Meta)

        # solve and compute error
        Annz[i,dindex] = nnz(A)
        u = A\b
        du = u - uexact(reshape(x, (2,num_nodes)))
        L2err[i, dindex] = sqrt( dot(vec(jac.*M), du.^2))
        maxerr[i, dindex] = maximum(abs.(du))
        println("degree = ",degree,": num_nodes = ",num_nodes,": error = ",
                L2err[i, dindex])

    end
end

# write the error to file for plotting
f = open("error-annulus-conv.dat", "w")
for d in axes(deg,1)
    for i in axes(num1d,1)
        print(f, dx[i,d], " ")
    end
    println(f)
    for i in axes(num1d,1)
        print(f, nodes[i,d], " ")
    end
    println(f)
    for i in axes(num1d,1)
        print(f, Annz[i,d], " ")
    end
    println(f)
    for i in axes(num1d,1)
        print(f, L2err[i,d], " ")
    end
    println(f)
    for i in axes(num1d,1)
        print(f, maxerr[i,d], " ")
    end
    println(f)
end
close(f)

end