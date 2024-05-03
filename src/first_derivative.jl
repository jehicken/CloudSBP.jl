
"""
Data for integrals on boundaries

Each boundary condition can be associated with a `BoundaryOperator`, which 
holds data needed to compute integrals over the relevant subset of the 
boundary.  This data is grouped by faces, since each face may require a 
different number of quadrature points or have a different sized stencil.  If 
`f` denotes the index of some face, then `xq_face[f][:,q]` is the `q`th 
quadrature point on face; `nrm_face[f][:,q]` is the quadrature-weighted outward 
facing normal at `xq_face[f][:,q]`; `dof_face[f][i]` is global DOF index 
associated with the local interpolation index `i`, and; `prj_face[f][q,:]` is 
the interpolation operator from the local degrees of freedom to quadrature node 
`q`.
"""
mutable struct BoundaryOperator{T}
    xq_face::Array{Matrix{T}}
    nrm_face::Array{Matrix{T}}
    dof_face::Array{Vector{Int}}
    prj_face::Array{Matrix{T}}
end

"""
    E = BoundaryOperator(T)

Returns a `BoundaryOperator` with empty arrays.
"""
function BoundaryOperator(T::Type)
    BoundaryOperator(Array{Matrix{T}}(undef,0), Array{Matrix{T}}(undef,0),
                     Array{Vector{Int}}(undef,0), Array{Matrix{T}}(undef,0))
end

"""
    xq, nrm, dof, prj = push_new_face!(bndry, Dim, num_nodes, num_quad)

Creates new memory at the end of the fields in the given `BoundaryOperator` and 
returns references to the newly created arrays.
"""
function push_new_face!(bndry::BoundaryOperator{T}, Dim, num_nodes, num_quad
                        ) where {T}
    push!(bndry.xq_face, zeros(Dim, num_quad))
    push!(bndry.nrm_face, zeros(Dim, num_quad))
    push!(bndry.dof_face, zeros(Int, num_nodes))
    push!(bndry.prj_face, zeros(num_quad, num_nodes))
    return bndry.xq_face[end], bndry.nrm_face[end], bndry.dof_face[end], 
        bndry.prj_face[end]
end


"""
Summation-by-parts first derivative operator

`S[d]` holds the skew-symmetric part for direction `d` and `E[:]` holds `BoundaryOperator`s that define the symmetric part.
"""
mutable struct SBP{T,Dim} #FirstDeriv{T, Dim}
    S::SVector{Dim, SparseMatrixCSC{T, Int64}}    
    #E::Matrix{SparseMatrixCSC{T, Int64}}
    bnd_pts::Array{Matrix{T}}
    bnd_nrm::Array{Matrix{T}}
    bnd_dof::Array{Vector{Int}}
    bnd_prj::Array{Matrix{T}}
end




"""
    add_face_to_boundary!(bndry, face, xc, degree)

Computes the quadrature points, normal vector, degrees of freedom, and 
interpolation operator for the face `face` and adds this data to the given 
`BoundaryOperator`, `bndry`.  `xc` are the locations of the nodes in the 
stencil of `face`, and `degree` determines the order of accuracy of the 
quadrature (`2*degree+1`) and the interpolation.
"""
function add_face_to_boundary!(bndry::BoundaryOperator{T}, face, xc, degree
                               ) where {T}
    cell = face.cell[1]
    Dim = size(xc,1)
    @assert( length(cell.data.points) == size(xc,2), 
            "face.cell[1] and xc are incompatible")
    x1d, w1d = lg_nodes(degree+1)
    num_nodes = length(cell.data.points)
    num_quad = length(w1d)^(Dim-1)
    wq_face = zeros(num_quad)
    xq_face, nrm, dof, prj = push_new_face!(bndry, Dim, num_nodes, num_quad)
    face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)
    build_interpolation!(prj, degree, xc, xq_face, cell.data.xref, cell.data.dx)
    for (q,w) in enumerate(wq_face) 
        nrm[abs(face.dir),q] = sign(face.dir)*w 
    end
    dof[:] = cell.data.points # just make a reference to points?
    return nothing
end

"""
    add_face_to_boundary!(bndry, face, xc, degree, levset [, fit_degree=degree])

This version of the method is for planar boundary faces that are cut by the
level-set geometry defined by the function `levset`.  The optional kwarg 
`fit_degree` indicates the degree of the Bernstein polynomials used to 
approximate the level-set within the Algoim library.
"""
function add_face_to_boundary!(bndry::BoundaryOperator{T}, face, xc, degree, 
                               levset; fit_degree::Int=degree
                               ) where {T}
    cell = face.cell[1]
    @assert( length(cell.data.points) == size(xc,2), 
            "face.cell[1] and xc are incompatible")
    wq_cut, xq_cut = cut_face_quad(face.boundary, face.dir, levset, degree+1,
                                     fit_degree=fit_degree)
    num_nodes = length(cell.data.points)
    num_quad = size(xq_cut,2)
    xq_face, nrm, dof, prj = push_new_face!(bndry, Dim, num_nodes, num_quad)
    xq_face[:,:] = xq_cut[:,:]
    build_interpolation!(prj, degree, xc, xq_face, cell.data.xref, cell.data.dx)
    for (q,w) in enumerate(wq_cut) 
        nrm[abs(face.dir),q] = sign(face.dir)*w 
    end
    dof[:] = cell.data.points # just make a reference to points?
    return nothing
end

"""
    add_face_to_boundary!(bndry, cell, xc, degree, levset, levset_grad!
                          [, fit_degree=degree])

This version of the method is for faces that are the intersection between the
level-set `levset(x)=0` and the cell `cell`.  In addition to the level-set, this
function needs a function that can compute the gradient of the level-set,
`levset_grad!(g, x)` which returns the gradient at `x` in the array `g`.  As
usual, `fit_degree` indicates the degree of the Bernstein polynomials used to 
approximate the level-set within the Algoim library.

**NOTE**: we use `2*degree+1` nodes in each direction for the surface
quadrature, since it must integrate the normal in addition to the flux.
"""
function add_face_to_boundary!(bndry::BoundaryOperator{T},
                               cell::Cell{Data, Dim, T, L}, xc, degree, levset,
                               levset_grad!; fit_degree::Int=degree
                               ) where {Data, Dim, T, L}
    @assert( length(cell.data.points) == size(xc,2), 
            "cell and xc are incompatible")
    surf_wts, surf_pts = cut_surf_quad(cell.boundary, levset, 2*degree+1,
                                       fit_degree=fit_degree)
    num_quad = size(surf_pts,2)
    if num_quad == 0
        # Algoim may determine the cell is not actually cut
        return nothing 
    end    
    num_nodes = length(cell.data.points)
    xq_face, nrm, dof, prj = push_new_face!(bndry, Dim, num_nodes, num_quad)
    xq_face[:,:] = surf_pts[:,:]
    build_interpolation!(prj, degree, xc, xq_face, cell.data.xref, cell.data.dx)
    nrm[:,:] = -surf_wts
    dof[:] = cell.data.points # just make a reference to points?

    # Algoim's quadrature weights are sometimes separated, in the sense that
    # each direction has distinct quadrature nodes; this is problematic for our
    # formulation, so correct that here.
    dphi = zeros(Dim)
    for q in axes(nrm,2)
        levset_grad!(dphi, surf_pts[:,q])
        dphi ./= -norm(dphi)
        dA = dot(nrm[:,q], dphi)
        nrm[:,q] = dphi*dA
    end
    return nothing
end

"""
    E = boundary_operators(bc_map, root, boundary_faces, xc, levset,
                           levset_grad!, degree [, fit_degree=degree])

Generates a set of `BoundaryOperator`s for each of the `2*Dim` planar boundary
as well as the immersed surface within the domain of the root cell `root`.
`bc_map` is a Dictionary that maps boundaries to boundary conditions; the key
used for a planar boundaries is its side index:

* 1 -- EAST (root.boundary.origin[1])
* 2 -- WEST (root.boundary.origin[1] + root.boundary.width[1])
* 3 -- SOUTH (root.boundary.origin[2])
* 4 -- NORTH (root.boundary.origin[2] + root.boundary.width[2])
* 5 -- BOTTOM (root.boundary.origin[3])
* 6 -- TOP (root.boundary.origin[3] + root.boundary.width[3])

The key for the immersed surface is the string "ib".  All the planar boundary
faces are stored in `boundary_faces`.  The DOF coordinates are given by `xc`.
The immersed surface is defined by `levset(x) = 0`, and the gradient of this
level set is `g` after calling `levset_grad!(g,x)`.  The formal polynomial
accuracy of the boundary operator is `degree`, while `fit_degree` gives the
degree of the Bernstein polynomial used by Algoim to fit `levset`.
"""
function boundary_operators(bc_map, root::Cell{Data, Dim, T, L}, boundary_faces,
                            xc, levset, levset_grad!, degree; fit_degree::Int=degree
                            ) where {Data, Dim, T, L}

    # Create a boundary operator for each unique BC
    bc_types = unique(values(bc_map))
    E = Dict( bc => BoundaryOperator(eltype(xc)) for bc in bc_types)

    # loop over all planar boundary faces.
    for face in boundary_faces
        if is_immersed(face)
            continue
        end
        di = abs(face.dir)
        side = 2*di + div(sign(face.dir) - 1,2)
        if is_cut(face)
            add_face_to_boundary!(E[bc_map[side]], face, 
                                  view(xc, :, face.cell[1].data.points), degree,
                                  levset, fit_degree=fit_degree)
        else
            add_face_to_boundary!(E[bc_map[side]], face, 
                                  view(xc, :, face.cell[1].data.points), degree)
        end
    end

    # loop over the non-planar boundary faces
    for cell in allleaves(root)
        if is_cut(cell)
            add_face_to_boundary!(E[bc_map["ib"]], cell, 
                                  view(xc, :, cell.data.points), degree,
                                  levset, levset_grad!, fit_degree=fit_degree)
        end
    end

    return E
end


function build_boundary_operator(root::Cell{Data, Dim, T, L}, boundary_faces, 
                                 xc, degree) where {Data, Dim, T, L}
    num_face = length(boundary_faces)
    bnd_nrm = Array{Matrix{T}}(undef, num_face)
    bnd_pts = Array{Matrix{T}}(undef, num_face)
    bnd_dof = Array{Vector{Int}}(undef, num_face)
    bnd_prj = Array{Matrix{T}}(undef, num_face)

    # find the maximum number of phi basis over all cells
    max_basis = 0
    for cell in allleaves(root)
        max_basis = max(max_basis, length(cell.data.points))
    end

    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    num_quad = length(w1d)^(Dim-1)
    wq_face = zeros(num_quad)
    
    #work = dgd_basis_work_array(degree, max_basis, length(wq_face), Val(Dim))
    #work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq_face))

    # loop over boundary faces 
    for (findex, face) in enumerate(boundary_faces)
        di = abs(face.dir)
        # get the Gauss points on face
        bnd_pts[findex] = zeros(Dim, num_quad)
        face_quadrature!(bnd_pts[findex], wq_face, face.boundary, x1d, w1d, di)
        # evaluate the basis functions on the face nodes; this defines prolong
        num_nodes = length(face.cell[1].data.points)
        bnd_prj[findex] = zeros(num_quad, num_nodes)
        build_interpolation!(bnd_prj[findex], degree,
                             view(xc, :, face.cell[1].data.points),
                             bnd_pts[findex], face.cell[1].data.xref,
                             face.cell[1].data.dx)

        #dgd_basis!(bnd_prj[findex], degree,
        #           view(points, :, face.cell[1].data.points),
        #           bnd_pts[findex], work, Val(Dim))
        # define the face normals
        bnd_nrm[findex] = zero(bnd_pts[findex])
        for q = 1:num_quad
            bnd_nrm[findex][di,q] = sign(face.dir)*wq_face[q]
        end
        # get the degrees of freedom 
        bnd_dof[findex] = deepcopy(face.cell[1].data.points)
    end
    return bnd_pts, bnd_nrm, bnd_dof, bnd_prj
end

# function build_boundary_operator(root::Cell{Data, Dim, T, L}, boundary_faces, 
#                                  points, degree) where {Data, Dim, T, L}
#     num_face = length(boundary_faces)
#     bnd_nrm = Array{Matrix{T}}(undef, num_face)
#     bnd_pts = Array{Matrix{T}}(undef, num_face)
#     bnd_dof = Array{Vector{Int}}(undef, num_face)
#     bnd_prj = Array{Matrix{T}}(undef, num_face)

#     # find the maximum number of phi basis over all cells
#     max_basis = 0
#     for cell in allleaves(root)
#         max_basis = max(max_basis, length(cell.data.points))
#     end

#     x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
#     num_quad = length(w1d)^(Dim-1)
#     wq_face = zeros(num_quad)
#     #work = dgd_basis_work_array(degree, max_basis, length(wq_face), Val(Dim))
#     work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq_face))

#     # loop over boundary faces 
#     for (findex, face) in enumerate(boundary_faces)
#         di = abs(face.dir)
#         # get the Gauss points on face
#         bnd_pts[findex] = zeros(Dim, num_quad)
#         face_quadrature!(bnd_pts[findex], wq_face, face.boundary, x1d, w1d, di)
#         # evaluate the basis functions on the face nodes; this defines prolong
#         num_basis = length(face.cell[1].data.points)
#         bnd_prj[findex] = zeros(num_quad, num_basis)
#         dgd_basis!(bnd_prj[findex], degree,
#                    view(points, :, face.cell[1].data.points),
#                    bnd_pts[findex], work, Val(Dim))
#         # define the face normals
#         bnd_nrm[findex] = zero(bnd_pts[findex])
#         for q = 1:num_quad
#             bnd_nrm[findex][di,q] = sign(face.dir)*wq_face[q]
#         end
#         # get the degrees of freedom 
#         bnd_dof[findex] = deepcopy(face.cell[1].data.points)
#     end
#     return bnd_pts, bnd_nrm, bnd_dof, bnd_prj
# end

function weak_differentiate!(dudx, u, di, sbp)
    fill!(dudx, 0)
    # first apply the skew-symmetric part of the operator 
    dudx[:] += sbp.S[di]*u 
    dudx[:] -= sbp.S[di]'*u
    # next apply the symmetric part of the operator 
    for (f, Pface) in enumerate(sbp.bnd_prj)
        nrm = sbp.bnd_nrm[f]
        for i = 1:size(Pface,2)
            row = sbp.bnd_dof[f][i]
            for j = 1:size(Pface,2)
                col = sbp.bnd_dof[f][j]
                for q = 1:size(Pface,1)                    
                    dudx[row] += 0.5*Pface[q,i]*nrm[di,q]*Pface[q,j]*u[col]
                end
            end
        end
    end
end