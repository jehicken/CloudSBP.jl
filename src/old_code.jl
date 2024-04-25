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