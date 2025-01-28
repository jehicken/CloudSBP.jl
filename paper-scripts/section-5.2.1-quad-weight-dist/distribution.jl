module Distribution
# This module produces files with the norm values, so that histograms can be
# created to investigate the distribution of the M_{ii} values.  It also 
# generates paraview files for visualizing the norms

using CloudSBP
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random

# Create repeatable "random" tests
Random.seed!(42)

# study parameters
Dim = 2
vel = ones(Dim)
deg = [1;2;3;4]
num1d = 16
pert_size = 0.25
growth = 0.1 # 4.0 for non-uniform, 0.1 for approximately uniform

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
    xc, num_nodes = get_points(nr, nt[, pert=0.1/nx, growth=5.0])

Generate quasi-uniform points by perturbing a uniform grid
"""
function get_points(nr, nt; pert::Float64=0.1, growth::Float64=5.0)
    @assert(Dim == 2)
    radius(xi) = (exp(growth*xi) - 1.0)/(exp(growth) - 1.0)
    dradius(xi) = growth*exp(growth*xi)/(exp(growth)-1.0)
    rin = 0.5
    rout = 1.0
    xc = zeros(Dim, nr*nt)
    xnd = reshape(xc, (Dim, nr, nt))
    dtheta = 2*pi/nt
    H_tol = zeros(size(xc,2))
    Hnd = reshape(H_tol, (nr, nt))
    for i in axes(xnd,2) # r loop
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

# generate mesh
xc, num_nodes, H_tol = get_points(num1d, 6*num1d, pert=pert_size, 
                                  growth=growth)

origin = SVector(ntuple(i -> -1.0, Dim))
widths = SVector(ntuple(i -> 2.0, Dim))
min_widths = ones(Dim)/(5*num1d)
mesh = CloudSBP.build_mesh(xc, widths, levset, min_widths, origin=origin)

for (dindex, degree) in enumerate(deg)
    
    CloudSBP.build_cell_stencils!(mesh, xc, 2*degree - 1)
    max_stencil, avg_stencil = CloudSBP.stencil_stats(mesh)
    println()
    println(repeat("=",80))
    println("degree = ",degree,": max_stencil = ",max_stencil,
            ": avg_stencil = ",avg_stencil)

    m = CloudSBP.calc_moments!(mesh.root, levset, max(2,2*degree-1), 2)

    # get the raw norms, before optimization 
    H = zero(H_tol)
    CloudSBP.diagonal_norm!(H, mesh.root, xc, 2*degree-1)
    norm_file = "norm_before_growth$(growth)_degree$(degree).dat"
    f = open(norm_file, "w")
    print(f, join(H, " "))
    close(f)

    # write the paraview file for the norm
    CloudSBP.point_data_vtk(xc, [abs.(H), sign.(H)], ["norm-mag", "norm-sign"],
                          filename="./norm_before_growth$(growth)_degree$(degree).vtu")

    # optimize
    H, success = CloudSBP.solve_norm!(mesh.root, xc, 2*degree-1, H_tol,
                                      verbose=true)
    if !success 
        error("Failed to find a positive norm")
    end
    norm_file = "norm_after_growth$(growth)_degree$(degree).dat"
    f = open(norm_file, "w")
    print(f, join(H, " "))
    close(f)

    # write the paraview file for the norm
    CloudSBP.point_data_vtk(xc, [abs.(H), sign.(H)], ["norm-mag", "norm-sign"],
    filename="./norm_after_growth$(growth)_degree$(degree).vtu")
    
    minH = minimum(H)
    @assert( minH > 0.0 )
    println("minimum(H) = ", minH)
    
end

end # module