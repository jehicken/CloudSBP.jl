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
    max_basis = 0
    for cell in allleaves(root)
        max_basis = max(max_basis, length(cell.data.points))
    end
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
