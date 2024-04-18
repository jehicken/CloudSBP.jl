

"""
Summation-by-parts first derivative operator

`S[d]` holds the skew-symmetric part for direction `d`
`E[b,d]` holds the symmetric part for boundary `b` and direction `d`
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
    rect = build_cell_face(dir, cell)

Construct a `HyperRectangle` for given cell on side `dir`.
"""
function cell_side_rect(dir, cell::Cell{Data, Dim, T, L}
                        ) where {Data, Dim, T, L}
    if dir > 0 
        origin = SVector(
            ntuple(i -> i == dir ? 
                   cell.boundary.origin[i] + cell.boundary.widths[i] :
                   cell.boundary.origin[i], Dim))
    else 
        origin = SVector(cell.boundary.origin)
    end
    widths = SVector(
        ntuple(i -> i == abs(dir) ? 0.0 : cell.boundary.widths[i], Dim))
    return HyperRectangle(origin, widths)
end

"""
    E = cell_symmetric_part(cell, xc, degree)

Returns the symmetric part of the first-derivative SBP operator for the uncut 
cell `cell`.  The point cloud associated with `cell` is `xc`, and the boundary 
operator is `2*degree` exact for boundary integrals.

**Note**: This version recomputes the 1D quadrature rule each time, and involves
several allocations.
"""
function cell_symmetric_part(cell::Cell{Data, Dim, T, L}, xc, degree
                             ) where {Data, Dim, T, L}
    @assert( length(cell.data.dx) > 0, "cell.data.dx is empty")
    @assert( length(cell.data.xref) > 0, "cell.data.xref is empty")
    num_nodes = size(xc,2)
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq_face = zeros(length(w1d)^(Dim-1))
    xq_face = zeros(Dim, length(wq_face))
    interp = zeros(length(wq_face), num_nodes)
    xref = cell.data.xref 
    dx = cell.data.dx
    E = zeros(num_nodes, num_nodes, Dim)
    for dir in ntuple(i -> i % 2 == 1 ? -div(i+1,2) : div(i,2), 2*Dim)
        rect = cell_side_rect(dir, cell)
        face_quadrature!(xq_face, wq_face, rect, x1d, w1d, abs(dir))
        build_interpolation!(interp, degree, xc, xq_face, xref, dx)
        for i in axes(interp,2)
            for j in axes(interp,2)
                for q in axes(interp,1)
                    E[i,j,abs(dir)] += interp[q,i] * wq_face[q] * interp[q,j] * sign(dir)
                end
            end
        end
    end
    return E
end

"""
    E = cell_symmetric_part(cell, xc, degree, levset [,geo_conserve=true])

Returns the symmetric part of the first-derivative SBP operator for the 
_possibly_ cut cell `cell`.  The point cloud associated with `cell` is `xc`, 
and the boundary operator is `2*degree` exact for boundary integrals.  If 
`geo_conserve` is `true`, then the geometric conservation law is enforced on 
the resulting E matrices.
"""
function cell_symmetric_part(cell::Cell{Data, Dim, T, L}, xc, degree, levset;
                             geo_conserve::Bool=true) where {Data, Dim, T, L}
    @assert( length(cell.data.dx) > 0, "cell.data.dx is empty")
    @assert( length(cell.data.xref) > 0, "cell.data.xref is empty")
    num_nodes = size(xc,2)
    xref = cell.data.xref 
    dx = cell.data.dx
    E = zeros(num_nodes, num_nodes, Dim)
    sumE = zeros(Dim)
    for dir in ntuple(i -> i % 2 == 1 ? -div(i+1,2) : div(i,2), 2*Dim)
        face = cell_side_rect(dir, cell)
        wq_face, xq_face = cut_face_quad(face, abs(dir), levset, degree+1,
                                         fit_degree=degree)
        interp = zeros(length(wq_face), num_nodes)
        build_interpolation!(interp, degree, xc, xq_face, xref, dx)
        for i in axes(interp,2)
            for j in axes(interp,2)
                for q in axes(interp,1)
                    E[i,j,abs(dir)] += interp[q,i] * wq_face[q] * interp[q,j] * sign(dir)
                end
            end
        end
        sumE[abs(dir)] += sum(wq_face)*sign(dir)
    end

    # at this point, all planar faces of cell have been accounted for; now deal 
    # with the level-set surface `levset(x) = 0` passing through the cell
    surf_wts, surf_pts = cut_surf_quad(cell.boundary, levset, degree+1,
                                       fit_degree=degree)

    if length(surf_wts) == 0
        # the cell was not actually cut, so the is nothing left to do but check
        for dir = 1:Dim
            @assert( abs(sumE[dir]) < 100*eps(), "geo. cons. law failed (1)" )
        end 
        return E
    end

    # NOTE: negative sign needed because of sign convention in algoim
    surf_wts .*= -1.0
    interp = zeros(size(surf_wts,2), num_nodes)
    build_interpolation!(interp, degree, xc, surf_pts, xref, dx)
    fac = 1/size(surf_wts,2)
    for dir = 1:Dim
        # correct for geometric conservation 
        if geo_conserve
            surf_wts[dir,:] -= fac*(sumE[dir] + sum(surf_wts[dir,:])) * 
                ones(size(surf_wts,2))
        end
        for i in axes(interp,2)
            for j in axes(interp,2)
                for q in axes(interp,1)
                    E[i,j,dir] += interp[q,i] * surf_wts[dir,q] * interp[q,j]
                end
            end
        end
        if geo_conserve
            @assert( abs(sum(E[:,:,dir])) < 10^(degree+1)*eps(),
                     "geo. cons. law failed (2)" )
        end
    end

    return E
end

"""
    S = cell_skew_part(cell, xc, degree, H, E)

Returns the skew-symmetric parts of SBP diagonal-norm operators for the element
`cell` based on the nodes `xc`.  The operator is exact for polynomials of total 
degree `degree`.  The diagonal norm for the cell is provided in the array `H`, 
which must be exact for degree `2*degree - 1` polynomials over `xc`.  Finally,
the symmetric part of the SBP operators must be provided in `E`.  Note that `E`
and the returned `S` are three dimensional arrays, with `E[:,:,d]` and
`S[:,:,d]` holding the operators for the direction `d`.
"""
function cell_skew_part(cell::Cell{Data, Dim, T, L}, xc, degree, H, E
                        ) where {Data, Dim, T, L}

    num_basis = binomial(Dim + degree, Dim)
    num_nodes = size(xc,2)

    xref = cell.data.xref 
    dx = cell.data.dx
    xc_trans = zero(xc)
    for I in CartesianIndices(xc)
        xc_trans[I] = (xc[I] - xref[I[1]])/dx[I[1]] - 0.5
    end

    V = zeros(num_nodes, num_basis)
    dV = zeros(num_nodes, num_basis, Dim)
    workc = zeros((Dim+1)*num_nodes)
    poly_basis!(V, degree, xc_trans, workc, Val(Dim))
    poly_basis_derivatives!(dV, degree, xc_trans, Val(Dim))

    # construct the linear system that the skew matrices must satisfy 
    num_skew_vars = div(num_nodes*(num_nodes-1),2)
    num_eqns = num_nodes*num_basis
    A = zeros(num_eqns, num_skew_vars)
    B = zeros(num_eqns, Dim)
    ptr = 0
    for k = 1:num_basis
        for row = 2:num_nodes 
            offset = div((row-1)*(row-2),2)
            for col = 1:row-1
                A[ptr+row,offset+col] += V[col,k]
                A[ptr+col,offset+col] -= V[row,k]
            end 
        end
        for d = 1:Dim
            # the factor of 1/dx[d] accounts for the transformation above
            B[ptr+1:ptr+num_nodes,d] = diagm(H)*dV[:,k,d]/dx[d] - 0.5*E[:,:,d]*V[:,k]
        end
        ptr += num_nodes
    end
    S = zeros(num_nodes, num_nodes, Dim)
    vals = zeros(num_skew_vars)
    for d = 1:Dim
        solve_min_norm!(vals, A', vec(B[:,d]))
        for row = 2:num_nodes 
            offset = div((row-1)*(row-2),2)
            for col = 1:row-1
                S[row,col,d] += vals[offset+col]
                S[col,row,d] -= vals[offset+col]
            end
        end
    end
    return S
end


function build_first_deriv(root::Cell{Data, Dim, T, L}, ifaces, xc, levset, degree
                           ) where {Data, Dim, T, L}
    num_nodes = size(xc, 2)
    #num_basis = binomial(Dim + degree, Dim)
    #@assert( size(H) == num_nodes, "H is inconsistent with xc")

    # find the maximum number of phi basis over all cells
    max_basis = 0
    for cell in allleaves(root)
        max_basis = max(max_basis, length(cell.data.points))
    end

    # set up arrays to store sparse matrix information
    rows = Array{Array{Int64}}(undef, Dim)
    cols = Array{Array{Int64}}(undef, Dim)
    Svals = Array{Array{T}}(undef, Dim)
    for d = 1:Dim
        rows[d] = Int[]
        cols[d] = Int[] 
        Svals[d] = T[]
    end

    # get arrays/data used for tensor-product quadrature 
    #x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    #num_quad = length(w1d)^Dim             
    #wq = zeros(num_quad)
    #xq = zeros(Dim, num_quad)
    #Vq = zeros(num_quad, num_basis)
    #dVq = zeros(num_quad, num_basis, Dim)
    #workq = zeros((Dim+1)*num_quad)

    #Eelem = zeros(max_basis, max_basis, Dim)
    #Selem = zeros(max_basis, max_basis, Dim)

    # set up the level-set function for passing to calc_cut_quad below
    mod_levset[] = levset
    safe_clevset = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))

    for (c, cell) in enumerate(allleaves(root))
        if cell.data.immersed
            continue
        end
        # get the nodes in this cell's stencil
        nodes = view(xc, :, cell.data.points)
        xref = cell.data.xref 
        dx = cell.data.dx
        moments = cell.data.moments
        Hcell = cell_quadrature(2*degree-1, nodes, moments, xref, dx, Val(Dim))

        if is_cut(cell)
            # this cell *may* be cut; use Saye's algorithm
            error("Not set up to handle cut cells yet...")
            Ecell = cell_symmetric_part(cell, nodes, degree, levset)

        else
            # this cell is not cut
            Ecell = cell_symmetric_part(cell, nodes, degree)
            #println("size(Hcell) = ",size(Hcell))
            #println("size(Ecell) = ",size(Ecell))
            #println("size(cell.data.points) = ",size(cell.data.points))
            Scell = cell_skew_part(cell, nodes, degree, Hcell, Ecell)
        end

        # Now load into sparse-matrix arrays
        for (i,row) in enumerate(cell.data.points)            
            for (j,col) in enumerate(cell.data.points[i+1:end])                
                for d = 1:Dim
                    if abs(Scell[i,i+j,d]) > 1e-13
                        append!(rows[d], row)
                        append!(cols[d], col)
                        append!(Svals[d], Scell[i,i+j,d])
                    end
                end
            end
        end
    end

    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq_face = zeros(length(w1d)^(Dim-1))
    xq_face = zeros(Dim, length(wq_face))
    num_face_quad = length(wq_face)
    work_left = zeros(max_basis*length(wq_face))
    work_right = zero(work_left)

    Sface = zeros(max_basis, max_basis)
    #Sface = view(Selem,:,:,1) # shallow copy

    # loop over interfaces
    for face in ifaces 
        cleft = face.cell[1]
        cright = face.cell[2]
        xc_left = view(xc, :, cleft.data.points)
        xc_right = view(xc, :, cright.data.points)
        num_basis_left = length(cleft.data.points)
        num_basis_right = length(cright.data.points)

        if is_immersed(face)
            # immersed faces do not contribute to derivative operator 
            continue
        elseif is_cut(face)
            error("Not set up to handle cut faces yet...")
        else
            face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)
            interp_left = reshape(view(work_left,1:num_basis_left*num_face_quad), 
                                  (num_face_quad, num_basis_left))
            build_interpolation!(interp_left, degree, xc_left, xq_face,
                                 cleft.data.xref, cleft.data.dx)
            interp_right = reshape(view(work_right,1:num_basis_right*num_face_quad), 
                                  (num_face_quad, num_basis_right))  
            build_interpolation!(interp_right, degree, xc_right, xq_face,
                                 cright.data.xref, cright.data.dx)
            fill!(Sface, zero(T))
            for i = 1:num_basis_left
                for j = 1:num_basis_right
                    for (q, wq) in enumerate(wq_face)
                        Sface[i,j] += interp_left[q,i] * interp_right[q,j] * wq
                    end
                    Sface[i,j] *= 0.5
                end
            end
        end

        # load into sparse-matrix arrays
        for (i,row) in enumerate(cleft.data.points)
            for (j,col) in enumerate(cright.data.points)
                if col == row continue end 
                if abs(Sface[i,j]) > 1e-13
                    append!(rows[face.dir], row)    
                    append!(cols[face.dir], col)
                    append!(Svals[face.dir], Sface[i,j])
                end
            end
        end
    end

    S = SVector(ntuple(d -> sparse(rows[d], cols[d], Svals[d]), Dim))
    return S
end

# function build_first_deriv(root::Cell{Data, Dim, T, L}, levset::LevelSet{Dim,T},
#                            points, degree) where {Data, Dim, T, L}

#     # Step 1: refine based on points (and levset?)

#     # Step 2: mark cut cells and cut faces 
#     mark_cut_cells!(root, levset)

#     # Step 3: set up arrays to store sparse matrix information
#     rows = Array{Array{Int64}}(undef, Dim)
#     cols = Array{Array{Int64}}(undef, Dim)
#     Svals = Array{Array{T}}(undef, Dim)
#     for d = 1:Dim
#         rows[d] = Int[]
#         cols[d] = Int[] 
#         Svals[d] = T[]
#     end

#     # Step 4: loop over cells and integrate volume integrals of bilinear form 
#     println("timing uncut_volume_integrate!...")
#     @time uncut_volume_integrate!(rows, cols, Svals, root, points, degree)
#     println("timing cut_volume_integrate!...")
#     @time cut_volume_integrate!(rows, cols, Svals, root, levset, points, degree)

#     # Step 5: loop over interfaces and integrate bilinear form 

#     # Step 6: loop over boundary faces and integrate bilinear form 

#     # Step 7: finalize construction 

#     # returns a FirstDeriv operator 
# end

"""
    uncut_volume_integrate!(rows, cols, Svals, root, points, degree)

Integrates the volume integral portion of the weak derivative operators over 
the cells identified as uncut and not immersed by their cooresponding data 
fields.  The resulting integrals are stored in the `rows`, `cols`, and `Svals` 
inputs; these inputs coorespond to the arrays that will later be used to 
construct sparse matrices for the Sx, Sy, and Sz skew symmetric operators.  The 
`root` input stores the tree mesh, the `points` array stores the DGD degree of 
freedom locations, and `degree` is the polynomial degree of exactness.
"""
function uncut_volume_integrate!(rows, cols, Svals, root::Cell{Data, Dim, T, L},
                                 points, degree) where {Data, Dim, T, L}
    @assert(length(rows) == length(cols) == length(Svals) == Dim, 
            "rows/cols/Svals inconsistent with Dim")

    # find the maximum number of phi basis over all cells
    max_basis = 0
    for cell in allleaves(root)
        max_basis = max(max_basis, length(cell.data.points))
    end

    # create some storage space 
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    phi = zeros(length(wq), max_basis, Dim+1)
    Selem = zeros(max_basis, max_basis, Dim)
    work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq))
    
    for cell in allleaves(root)
        if cell.data.cut || cell.data.immersed
            # do not integrate cells that have may be cut or immersed 
            continue 
        end 
        # get the Gauss points on cell
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        # phi[:,:,:] is used to both the DGD basis and its derivatves at xq
        dgd_basis!(phi, degree, view(points, :, cell.data.points), xq, 
                   cell.data.xref, cell.data.dx, work, Val(Dim))
        num_basis = length(cell.data.points)
        fill!(Selem, zero(T))
        for i = 1:num_basis
            for j = i+1:num_basis
                # loop over the differentiation directions
                for d = 1:Dim                    
                    for q = 1:length(wq)                        
                        Selem[i,j,d] += 0.5*(phi[q,i,1] * phi[q,j,1+d] - 
                                             phi[q,j,1] * phi[q,i,1+d]) * wq[q]
                    end
                end
            end 
        end
        # Now load into sparse-matrix arrays
        for i = 1:length(cell.data.points)
            row = cell.data.points[i]
            for j = i+1:length(cell.data.points)
                col = cell.data.points[j]
                for d = 1:Dim
                    if abs(Selem[i,j,d]) > 1e-13
                        append!(rows[d], row)
                        append!(cols[d], col)
                        append!(Svals[d], Selem[i,j,d])
                    end
                end
            end
        end        
    end
    return nothing
end

"""
    cut_volume_integrate!(rows, cols, Svals, root, levset, points, degree)

Integrates the volume integral portion of the weak derivative operators over 
the cells identified as cut by the appropriate data field.  The resulting 
integrals are stored in the `rows`, `cols`, and `Svals` inputs; these inputs 
correspond to the arrays that will later be used to construct sparse matrices 
for the Sx, Sy, and Sz skew symmetric operators. The `root` input stores the 
tree mesh, `levset` is the level-set data structure, `points` stores the DGD 
degree of freedom locations, and `degree` is the polynomial degree of exactness.
"""
function cut_volume_integrate!(rows, cols, Svals,
                               root::Cell{Data, Dim, T, L},
                               levset::LevelSet{Dim,T}, points, degree
                               ) where {Data, Dim, T, L}
    @assert(length(rows) == length(cols) == length(Svals) == Dim, 
            "rows/cols/Svals inconsistent with Dim")

    # find the maximum number of phi basis over all cells
    max_basis = 0
    for cell in allleaves(root)
        max_basis = max(max_basis, length(cell.data.points))
    end
    
    mod_levset[] = levset
    safe_clevset = @safe_cfunction( 
        x -> evallevelset(x, mod_levset[]), Cdouble, (Vector{Float64},))

    # create some storage space    
    Selem = zeros(max_basis, max_basis, Dim)

    for cell in allleaves(root)
        if !cell.data.cut || cell.data.immersed
            # do not integrate cells that have been confirmed uncut or immersed
            continue
        end
        println("Here I am!")

        # get the quadrature rule for this cell 
        wq, xq, surf_wts, surf_pts = calc_cut_quad(cell.boundary, safe_clevset,
                                                     degree+1, 
                                                     fit_degree=2*degree)
        # phi[:,:,:] is used to both the DGD basis and its derivatves at xq
        phi = zeros(length(wq), max_basis, Dim+1)
        work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq))
        dgd_basis!(phi, degree, view(points, :, cell.data.points), xq, 
                   cell.data.xref, cell.data.dx, work, Val(Dim))
        num_basis = length(cell.data.points)
        fill!(Selem, zero(T))
        for i = 1:num_basis
            for j = i+1:num_basis
                # loop over the differentiation directions
                for d = 1:Dim                    
                    for q = 1:length(wq)                        
                        Selem[i,j,d] += 0.5*(phi[q,i,1] * phi[q,j,1+d] - 
                                             phi[q,j,1] * phi[q,i,1+d]) * wq[q]
                    end
                end
            end 
        end
        # Now load into sparse-matrix arrays
        for i = 1:length(cell.data.points)
            row = cell.data.points[i]
            for j = i+1:length(cell.data.points)
                col = cell.data.points[j]
                for d = 1:Dim
                    if abs(Selem[i,j,d]) > 1e-13
                        append!(rows[d], row)
                        append!(cols[d], col)
                        append!(Svals[d], Selem[i,j,d])
                    end
                end
            end
        end        
    end
    return nothing 
end

function build_first_deriv(root::Cell{Data, Dim, T, L}, faces, points, degree
                           ) where {Data, Dim, T, L}
    # Initialize the rows, cols, and vals for sparse matrix storage of S
    rows = Array{Array{Int64}}(undef, Dim)
    cols = Array{Array{Int64}}(undef, Dim)
    Svals = Array{Array{T}}(undef, Dim)
    for d = 1:Dim
        rows[d] = zeros(Int, (0))
        cols[d] = zeros(Int, (0))
        Svals[d] = zeros(T, (0))
    end

    # find the maximum number of phi basis over all cells
    max_basis = 0
    for cell in allleaves(root)
        max_basis = max(max_basis, length(cell.data.points))
    end

    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    phi = zeros(length(wq), max_basis, Dim+1)
    Selem = zeros(max_basis, max_basis, Dim)
    #work = dgd_basis_work_array(degree, max_basis, length(wq), Val(Dim))
    work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq))
    println("timing volume loop...")
    @time begin
    for cell in allleaves(root)
        # get the Gauss points on cell
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        # phi[:,:,:] holds both the DGD basis and its derivatves at xq
        #println("timing dgd_basis!...")
        dgd_basis!(phi, degree, view(points, :, cell.data.points), xq,
                   cell.data.xref, cell.data.dx, work, Val(Dim))
        num_basis = length(cell.data.points)
        #Selem = zeros(num_basis, num_basis, Dim)
        fill!(Selem, zero(T))
        #println("timing Selem loop...")
        #@time begin 

        for i = 1:num_basis
            for j = i+1:num_basis
                # loop over the differentiation directions
                for d = 1:Dim
                    #Sij, Sji = 0.0, 0.0
                    for q = 1:length(wq)
                        #Sij += 0.5*phi[q,i,1] * phi[q,j,1+d] * wq[q]
                        #Sji += 0.5*phi[q,j,1] * phi[q,i,1+d] * wq[q]
                        Selem[i,j,d] += 0.5*(phi[q,i,1] * phi[q,j,1+d] - 
                                             phi[q,j,1] * phi[q,i,1+d]) * wq[q]
                    end
                    #Selem[i,j,d] += 0.5*(Sij - Sji)
                    #Selem[j,i,d] += 0.5*(Sji - Sij) # might not need this
                end
            end 
        end

        #end # @time begin
        #for d = 1:Dim 
        #    @assert( norm(Selem[:,:,d] + Selem[:,:,d]') < 1e-12 )
        #end
        # Now load into sparse-matrix arrays
        for i = 1:length(cell.data.points)
            row = cell.data.points[i]
            for j = i+1:length(cell.data.points)
                col = cell.data.points[j]
                for d = 1:Dim
                    if abs(Selem[i,j,d]) > 1e-13
                        append!(rows[d], row)
                        append!(cols[d], col)
                        append!(Svals[d], Selem[i,j,d])
                    end
                end
            end
        end        
    end
    end

    wq_face = zeros(length(w1d)^(Dim-1))
    xq_face = zeros(Dim, length(wq_face))
    phi_left = zeros(length(wq_face), max_basis)
    phi_right = zero(phi_left)
    Sface = view(Selem,:,:,1) # shallow copy

    # loop over interior faces 
    println("timing face loop...")
    @time begin
    for face in faces 
        # get the Gauss points on face and evaluate the basis functions there
        face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)
        cell = face.cell[1]
        dgd_basis!(phi_left, degree, view(points, :, cell.data.points),
                   xq_face, cell.data.xref, cell.data.dx, work, Val(Dim))
        cell = face.cell[2]
        dgd_basis!(phi_right, degree, view(points, :, cell.data.points),
                   xq_face, cell.data.xref, cell.data.dx, work, Val(Dim))
        num_basis_left = length(face.cell[1].data.points)
        num_basis_right = length(face.cell[2].data.points)
        #Sface = zeros(num_basis_left, num_basis_right)
        fill!(Sface, zero(T))
        for i = 1:num_basis_left
            for j = 1:num_basis_right
                for q = 1:length(wq_face)
                    Sface[i,j] += phi_left[q,i] * phi_right[q,j] * wq_face[q]
                end
                Sface[i,j] *= 0.5
            end
        end
        # Now load into sparse-matrix arrays
        for i = 1:num_basis_left
            row = face.cell[1].data.points[i]
            for j = 1:num_basis_right
                col = face.cell[2].data.points[j]
                if col == row continue end 
                if abs(Sface[i,j]) > 1e-13
                    append!(rows[face.dir], row)    
                    append!(cols[face.dir], col)
                    append!(Svals[face.dir], Sface[i,j])
                end
            end
        end
    end
    end

    S = SVector(ntuple(d -> sparse(rows[d], cols[d], Svals[d]), Dim))
    #S = SVector{Dim,SparseMatrixCSC{T, Int64}}(undef)
    #for d = 1:Dim 
    #    S[d] = sparse(rows[d], cols[d], Svals[d])
    #end
    return S
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