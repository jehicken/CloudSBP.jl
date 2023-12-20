

"""
Summation-by-parts first derivative operator

`S[d]` holds the skew-symmetric part for direction `d`
`E[b,d]` holds the symmetric part for boundary `b` and direction `d`
"""
mutable struct FirstDeriv{T, Dim}
    S::SVector{Dim, SparseMatrixCSC{T, Int64}}    
    E::Matrix{SparseMatrixCSC{T, Int64}}
    #bnd_pts::Array{Matrix{T}}
    #bnd_nrm::Array{Matrix{T}}
    #bnd_dof::Array{Vector{Int}}
    #bnd_prj::Array{Matrix{T}}
end

function build_first_deriv(root::Cell{Data, Dim, T, L}, levset::LevelSet{Dim,T},
                           points, degree) where {Data, Dim, T, L}

    # Step 1: refine based on points (and levset?)

    # Step 2: mark cut cells and cut faces 
    mark_cut_cells!(root, levset)

    # Step 3: set up arrays to store sparse matrix information
    rows = Array{Array{Int64}}(undef, Dim)
    cols = Array{Array{Int64}}(undef, Dim)
    Svals = Array{Array{T}}(undef, Dim)
    for d = 1:Dim
        rows[d] = Int[]
        cols[d] = Int[] 
        Svals[d] = T[]
    end

    # Step 4: loop over cells and integrate volume integrals of bilinear form 
    println("timing uncut_volume_integrate!...")
    @time uncut_volume_integrate!(rows, cols, Svals, root, points, degree)
    println("timing cut_volume_integrate!...")
    @time cut_volume_integrate!(rows, cols, Svals, root, levset, points, degree)

    # Step 5: loop over interfaces and integrate bilinear form 

    # Step 6: loop over boundary faces and integrate bilinear form 

    # Step 7: finalize construction 

    # returns a FirstDeriv operator 
end

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
        dgd_basis!(phi, degree, view(points, :, cell.data.points), xq, work,
                   Val(Dim))
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
        dgd_basis!(phi, degree, view(points, :, cell.data.points), xq, work,
                   Val(Dim))
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
        dgd_basis!(phi, degree, view(points, :, cell.data.points), xq, work, Val(Dim))
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
        dgd_basis!(phi_left, degree, view(points, :, face.cell[1].data.points),
                   xq_face, work, Val(Dim))
        dgd_basis!(phi_right, degree, view(points, :, face.cell[2].data.points),
                   xq_face, work, Val(Dim))
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
                                 points, degree) where {Data, Dim, T, L}
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
    work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq_face))

    # loop over boundary faces 
    for (findex, face) in enumerate(boundary_faces)
        di = abs(face.dir)
        # get the Gauss points on face
        bnd_pts[findex] = zeros(Dim, num_quad)
        face_quadrature!(bnd_pts[findex], wq_face, face.boundary, x1d, w1d, di)
        # evaluate the basis functions on the face nodes; this defines prolong
        num_basis = length(face.cell[1].data.points)
        bnd_prj[findex] = zeros(num_quad, num_basis)
        dgd_basis!(bnd_prj[findex], degree,
                   view(points, :, face.cell[1].data.points),
                   bnd_pts[findex], work, Val(Dim))
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