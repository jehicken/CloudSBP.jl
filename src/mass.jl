"""
    M = mass_matrix(root, points, degree)

Returns the DGD mass matrix for degree `degree` based on the mesh in the tree 
`root` and the centers in `points`.
"""
function mass_matrix(root::Cell{Data, Dim, T, L}, points, degree
                     ) where {Data, Dim, T, L}
    rows = zeros(Int, (0))
    cols = zeros(Int, (0))                        
    Mvals = zeros(T, (0))
    # find the maximum number of phi basis over all cells
    max_basis = max_leaf_stencil(root)
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    phi = zeros(length(wq), max_basis)
    Melem = zeros(T, max_basis, max_basis)
    work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq))
    for cell in allleaves(root)
        # get the Gauss points on cell
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        # phi[:,:,:] holds the DGD basis at xq 
        dgd_basis!(phi, degree, view(points, :, cell.data.points), xq, work,
                   Val(Dim))
        num_basis = length(cell.data.points)
        fill!(Melem, zero(T))
        for i = 1:num_basis
            for j = 1:num_basis
                # add contributions to mass matrix 
                for q = 1:size(wq,1)
                    Melem[i,j] += phi[q,i] * phi[q,j] * wq[q]
                end
            end 
        end
        @assert( norm(Melem - Melem') < 1e-12 )
        # Now load into sparse-matrix array
        for i = 1:length(cell.data.points)
            row = cell.data.points[i]
            for j = 1:length(cell.data.points)
                col = cell.data.points[j]
                if abs(Melem[i,j]) > 1e-13
                    append!(rows, row)
                    append!(cols, col)
                    append!(Mvals, Melem[i,j])
                end
            end
        end  
    end
    return sparse(rows, cols, Mvals)
end

"""
    mass_row_sums!(lumped_mass, root, points, degree)

Fills the array `lumped_mass` with the sum of the rows (or columns) of the 
symmetric DGD mass matrix of degree `degree` based on the cloud `points`.
"""
function mass_row_sums!(lumped_mass, root::Cell{Data, Dim, T, L}, points, degree
                        ) where {Data, Dim, T, L}
    @assert( length(lumped_mass == size(points,2),
            "lumped_mass inconsistent with points" )
    fill!(lumped_mass, zero(eltype(lumped_mass)))
    # find the maximum number of phi basis over all cells
    max_basis = max_leaf_stencil(root)
    x1d, w1d = lg_nodes(degree+1) # could also use lgl_nodes
    wq = zeros(length(w1d)^Dim)
    xq = zeros(Dim, length(wq))
    phi = zeros(length(wq), max_basis)
    work = DGDWorkSpace{T,Dim}(degree, max_basis, length(wq))
    for cell in allleaves(root)
        # get the Gauss points on cell
        quadrature!(xq, wq, cell.boundary, x1d, w1d)
        # phi[:,:,:] holds the DGD basis at xq 
        dgd_basis!(phi, degree, view(points, :, cell.data.points), xq, work,
                   Val(Dim))
        num_basis = length(cell.data.points)
        fill!(Melem, zero(T))
        for i = 1:num_basis
            # add contributions to element row sums
            for q = 1:size(wq,1)
                lumped_mass[cell.data.points[i]] += phi[q,i] * wq[q]
            end
        end
    end
    return nothing
end

"""
    obj = mass_matrix_obj(root, xc, xc_init, dist_ref, D_tol, mu, degree)

Compute the objective that seeks to minimize the change in the node locations
while ensuring a positive lumped mass matrix.  `root` is the mesh, `xc` are the 
nodes being varied, and `xc_init` are the initial node locations. `dist_ref` are
reference lengths, and `D_tol` is an array of tolerances for the lumped mass 
value at each node; that is, `sum_{j} M[i,j] >= D_tol[i]` for the constraint to 
be satisfied.  Finally, `mu` scales the regularization term, and `degree` is 
the target exactness of the rule.
"""
function mass_matrix_obj(root::Cell{Data, Dim, T, L}, points, points_init, 
                         dist_ref, D_tol, mu, degree)  where {Data, Dim, T, L})
    num_nodes = size(xc, 2)
    # compute the penalty based on change from points_init
    phi = 0.0
    for i = 1:num_nodes 
        dist = 0.0
        for d = 1:Dim 
            dist += (points[d,i] - points_init[d,i])^2
        end 
        phi += 0.5*mu*dist/(dist_ref[i]^2)
    end
    # compute the lumped mass matrix based on points 
    H = zeros(eltype(points), num_nodes)
    mass_row_sums!(H, root, points, degree)
        # and add the penalties
    for i = 1:num_nodes
        if real(H[i]) < D_tol[i]
            phi += 0.5*(H[i]/D_tol[i] - 1)^2
        end 
    end
    return phi
end

