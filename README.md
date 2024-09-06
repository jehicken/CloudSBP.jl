# CloudSBP

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jehicken.github.io/CloudSBP.jl/)
[![Build Status](https://github.com/jehicken/CloudSBP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jehicken/CloudSBP.jl/actions/workflows/CI.yml?query=branch%3Amain)

This Julia package is a research code for constructing summation-by-parts (SBP) finite-difference operators on point clouds over complex geometries.  It is the library that implements the algorithm and produced the results in the following paper:

> Jason Hicken, Ge Yan, and Sharanjeet Kaur, _"Constructing stable, high-order finite-difference operators on point clouds over complex geometries,"_ submitted (see also this [preprint](http://arxiv.org/abs/2409.00809) on arxiv)

The implementation of the construction algorithm is not particularly efficient at this time; however, once the SBP operators are constructed as sparse arrays, they should be reasonably efficient to use.

## Installation

We are working toward registering the package, but, in the meantime, you will have the follow the instructions [here](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages) in order to use it.

## Documentation

As a research code, CloudSBP does not have a lot of documentation.  The `doc` badge will bring you to the `Documenter.jl` generated documentation for each function.  The test files may provide some insights as well.

The example below might be useful.  It is similar to the code used to generate the data for L2 and functional error studies in the paper above.  If you copy this into a file (e.g. `example.jl`), you can run it by using `include("example.jl")` from the Julia REPL.

```julia
module AccuracyStudyAnnulus

using CloudSBP
using StaticArrays: SVector
using LinearAlgebra
using Random
using SparseArrays
using Statistics

# for repeatable "random" tests
Random.seed!(42)

# study parameters
Dim = 2           # The spatial dimension; this example expects 2d
vel = ones(Dim)   # The velocity field
deg = [1;2;3;4]   # The polynomial degrees of the SBP operator
num1d = [6; 12; 24; 48] # The number of nodes along one dimension for node seq.
num_sample = 1    # The number of samples (10 samples were used in the paper)
pert_size = 0.25  # Perturbation relative magnitude used for nodes
growth = 0.1      # 0.1 for approximately uniform, 4.0 for non-uniform

# origin and widths define the size and location of the root cell in the mesh
origin = SVector(ntuple(i -> -1.0, Dim))
widths = SVector(ntuple(i -> 2.0, Dim))

# All boundaries have characteristic BCs applied to them
# See how this information is used in the `build_system` function below
bc_map = Dict{Any,String}("ib" => "upwind") # "ib" denotes immersed boundaries
for di = 1:Dim # Mark the root cell's boundaries; unnecessary for this case
    bc_map[di*2-1] = "upwind"
    bc_map[di*2] = "upwind"
end

# Define the level-set and its gradient.
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

# Define the exact solution; we use the method of manufactured solutions
uexact(x::AbstractVector) = exp(sum(x))
dudx(x::AbstractVector) = Dim*exp(sum(x))
uexact(x::AbstractMatrix) = exp.(sum(x,dims=1))'
dudx(x::AbstractMatrix) = Dim*exp.(sum(x,dims=1))'

"""
    f = functional(x)

Defines the integrand for the functional error study.
"""
function functional(x)
    r = norm(x)
    return exp(r)/r
end
fun_exact = 2*pi*(exp(1) - exp(0.5))

"""
    xc, num_nodes, H_tol = get_points(nr, nt[, pert=0.1, growth=4.0])

Generate points for the annulus by first creating the points in polar 
coordinates and then mapping them to Cartesian coordinates.  `nr` and `nt` 
indicate the number of nodes in the radial and angular directions, 
respectively.  The kwarg `pert` indicates the relative size of the perturbation
applied to the nodes, and `growth` is used to control clustering of nodes
toward the inner radius.  The method returns the points, `xc`, the number of
nodes, `num_nodes`, and the tolerance for the diagonal norm, `H_tol`.
"""
function get_points(nr, nt; pert::Float64=0.1, growth::Float64=4.0)
    @assert(Dim == 2)
    radius(xi) = (exp(growth*xi) - 1.0)/(exp(growth) - 1.0)
    dradius(xi) = growth*exp(growth*xi)/(exp(growth)-1.0)
    rin = 0.5
    rout = 1.0
    xc = zeros(Dim, nr*nt)
    xnd = reshape(xc, (Dim, nr, nt)) # helpful for nested loop
    dtheta = 2*pi/nt
    H_tol = zeros(size(xc,2))
    Hnd = reshape(H_tol, (nr, nt))
    for i in axes(xnd,2) # radial loop
        dr = dradius((i-1)/nr + 0.5/nr)/nr
        for j in axes(xnd,3) # theta loop
            xi = radius((i-1)/nr + 0.5/nr + pert*(2*rand()-1)/nr)
            r = (1-xi)*rin + xi*rout 
            theta = (j-1)*dtheta + 0.5*dtheta + pert*dtheta*(2*rand()-1)
            xnd[1,i,j] = r*cos(theta)
            xnd[2,i,j] = r*sin(theta)
            Hnd[i,j] = 0.1*dr*dtheta*r
        end
    end
    return xc, nr*nt, H_tol
end

"""
    A, b = build_system(sbp, diss, xc [, diss_coeff=0.25])

Returns the linear system corresponding to the SBP discretization using the 
operator `sbp`, dissipation `diss`, and the nodes `xc`.  Returns the system 
matrix `A` and the right-hand side `b`.  The kwarg `diss_coeff` scales the 
dissipation added to the system.
"""
function build_system(sbp, diss, xc; diss_coeff::Float64=0.25)    
    # Use the SBP operator to size the linear system
    num_nodes = length(sbp.H)
    A = similar(sbp.S[1] + sbp.S[1]')
    fill!(A, zero(Float64))
    b = zeros(num_nodes)
    
    # first apply the skew-symmetric part of the operator
    for di = 1:Dim 
        A += vel[di]*(sbp.S[di] - sbp.S[di]')
    end
    
    # now loop over the boundary and apply numerical flux function as needed
    for (bc_type, bndry) in sbp.E
        if bc_type == "upwind"
            for (xq, nrm, dof, prj) in zip(bndry.xq_face, bndry.nrm_face, 
                bndry.dof_face, bndry.prj_face)
                for i in axes(prj,2) # loop over nodes that are impacted
                    row = dof[i]
                    for j in axes(prj,2) # loop over nodes that influence
                        col = dof[j]
                        for q in axes(prj,1) # loop over face quadrature points
                            velnrm = dot(vel, nrm[:,q])
                            if velnrm < 0.0
                                # This is inflow 
                                A[row,col] -= 0.5*prj[q,i]*velnrm*prj[q,j]
                            else
                                # This is outflow 
                                A[row,col] += 0.5*prj[q,i]*velnrm*prj[q,j]
                            end
                        end
                    end
                    # Add to the right-hand side array
                    for q in axes(prj,1)
                        velnrm = dot(vel, nrm[:,q])
                        if velnrm < 0.0
                            # This is inflow
                            b[row] -= prj[q,i]*velnrm*uexact(xq[:,q])
                        end
                    end
                end
            end
        end
    end
    
    # apply the dissipation
    A += diss_coeff*(diss.R_left' - diss.R_right')*spdiagm(diss.w_face)*(diss.R_left - diss.R_right)
    
    # finally, add the MMS terms
    b += sbp.H .* dudx(xc)
    
    return A, b
end

# follow arrays are used to store various statistics
L2err = zeros(length(num1d), num_sample, length(deg))
dx = zero(L2err)
Hdx = zero(L2err)
minH = zero(L2err)
maxerr = zero(L2err)
funerr = zero(L2err)

# loop over samples
for k = 1:num_sample 

    # loop over mesh sizes
    for (i, nx) in enumerate(num1d)
        # generate the nodes and mesh for this sample 
        xc, num_nodes, H_tol = get_points(nx, 6*nx, pert=pert_size, 
                                          growth=growth)
        dx[i, k, :] .= 1/nx
        min_widths = ones(Dim)/(20*nx)
        mesh = CloudSBP.build_mesh(xc, widths, levset, min_widths,
                                   origin=origin)

        # loop over the polynomial degrees
        for (dindex, degree) in enumerate(deg)
            println()
            println(repeat("=",80))
            CloudSBP.build_cell_stencils!(mesh, xc, 2*degree - 1)            
            max_stencil, avg_stencil = CloudSBP.stencil_stats(mesh)
            println("degree = ",degree,": max_stencil = ",max_stencil,
                    ": avg_stencil = ",avg_stencil)            
            m = CloudSBP.calc_moments!(mesh.root, levset, max(2,2*degree-1), 2)

            # find diagonal norm
            H, success = CloudSBP.solve_norm!(mesh.root, xc, 2*degree-1, H_tol,
                                              verbose=true)
            minH[i, k, dindex] = minimum(H)
            println("minimum(H) = ", minH[i, k, dindex])

            if !success
                # failed to find a positive definite diagonal mass matrix
                println("The LP optimization failed")
                Hdx[i, k, dindex] = NaN
                L2err[i, k, dindex] = NaN
                maxerr[i, k, dindex] = NaN
                continue
            end
            @assert( minH[i, k, dindex] > 0.0 )

            # build the SBP operators, the dissipation, and the linear system
            sbp = CloudSBP.build_first_derivative(mesh, bc_map, xc, levset, 
                                                  levset_grad!, degree,
                                                  fit_degree=2)
            diss = CloudSBP.build_face_dissipation(mesh.ifaces, xc, degree, 
                                                   levset, fit_degree=2)
            A, b = build_system(sbp, diss, xc)

            # solve and compute the SBP-approximation of the L2 error 
            u = A\b 
            du = u - uexact(xc)
            L2err[i, k, dindex] = sqrt(dot(du,sbp.H.*du))
            Hdx[i, k, dindex] = mean(H)^(1/Dim)
            maxerr[i, k, dindex] = maximum(abs.(du))
            println("degree = ",degree,": num_nodes = ",num_nodes,": error = ",
                    L2err[i, k, dindex])

            # compute the functional error
            fun = 0.0
            for i in axes(xc, 2)
                fun += functional(xc[:,i])*H[i]
            end
            funerr[i, k, dindex] = abs(fun - fun_exact)
            println("fun error = ",funerr[i, k, dindex])

            if degree == 10 && i == 1 && k == 1
                # select `degree` and `i` to write error to VTK file
                CloudSBP.points_vtk(xc, filename="points")
                du = abs.(du)
                CloudSBP.output_vtk(mesh.root, xc, degree, du, filename="error")
                error("Wrote error file; stopping convergence study!!!")
            end

        end
    end
end

# output the statistics to a file; note that the data is written in a different 
# order than the arrays' dimensions.
f = open("error-annulus.dat", "w")
for d in axes(deg,1)
    for k in 1:num_sample 
        for i in axes(num1d,1)
            print(f, dx[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, Hdx[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, minH[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, L2err[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, maxerr[i,k,d], " ")
        end
        println(f)
        for i in axes(num1d,1)
            print(f, funerr[i,k,d], " ")
        end
        println(f)
    end
end
close(f)

end
```


