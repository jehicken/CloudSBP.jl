
"""
    xplot, uplot = output_pyplot(root, xc, degree, u [, num_pts=degree+1])

Outputs "meshgrid" type data for each cell in `root` for plotting the contours 
of the state `u` using `PyPlot` (i.e. matplotlib).  The solution is of degree 
`degree` and `num_pts` samples are used in each direction on each cell.
"""
function output_pyplot(root::Cell{Data, Dim, T, L}, xc, degree, u;
                       num_pts=degree+1) where {Data, Dim, T, L}
    
    num_cell = 0
    for cell in allleaves(root)
        if !is_immersed(cell)
            num_cell += 1
        end
    end
    xplot = Vector{Matrix{T}}(undef, num_cell)
    uplot = Vector{Vector{T}}(undef, num_cell)
    
    ptr = 1
    for cell in allleaves(root)
        if is_immersed(cell) continue end 
        rect = cell.boundary
        xplot[ptr] = zeros(Dim, num_pts^Dim)
        xnd = reshape(xplot[ptr], (Dim, ntuple(i -> num_pts, Dim)...))
        for I in CartesianIndices(xnd)
            # I[1] is the coordinate, so I[I[1] + 1] is the index for that coord
            xnd[I] = rect.origin[I[1]] + ((I[I[1]+1] - 1)/(num_pts-1))*rect.widths[I[1]]
        end
        
        # interpolate the solution to the print nodes
        xc_cell = view(xc, :, cell.data.points)
        interp = zeros(size(xplot[ptr],2), length(cell.data.points))
        build_interpolation!(interp, degree, xc_cell, xplot[ptr],
        cell.data.xref, cell.data.dx)
        uplot[ptr] = interp*u[cell.data.points]
        ptr += 1
    end
    return xplot, uplot
end

function output_vtk(root::Cell{Data, 2, T, L}, xc, degree, u;
                    num_pts=degree+1, filename="solution") where {Data, T, L}
    
    # count number of cells that are not immersed
    Dim = 2
    num_cell = 0
    for cell in allleaves(root)
        if !is_immersed(cell)
            num_cell += 1
        end
    end

    # generate the vtk points, cells, and data
    coords = zeros(Dim, (2^Dim)*num_cell)
    data = zeros((2^Dim)*num_cell)
    vtk_cells = MeshCell{VTKCellType, Vector{Int}}[]
    ptr = 0
    for cell in allleaves(root)
        if is_immersed(cell) continue end         
        coords[:, ptr+1:ptr+2^Dim] = hcat(collect(vertices(cell.boundary))...)
        v = [ptr+1; ptr+2; ptr+4; ptr+3]        
        push!(vtk_cells, MeshCell(VTKCellTypes.VTK_QUAD, v))

        # interpolate the solution to the print nodes
        xc_cell = view(xc, :, cell.data.points)
        interp = zeros(2^Dim, length(cell.data.points))
        build_interpolation!(interp, degree, xc_cell, coords[:,ptr+1:ptr+2^Dim],
                             cell.data.xref, cell.data.dx)
        data[ptr+1:ptr+2^Dim] = interp*u[cell.data.points]

        ptr += 2^Dim
    end
    
    # create the vtk file of the solution
    file = vtk_grid(filename, coords, vtk_cells) do vtk
        # add datasets...
        vtk_point_data(vtk, data, "scalar solution")
    end

    # create the vtk file of the cloud points 
    vtk_points = MeshCell{VTKCellType, Vector{Int}}[]
    for i in axes(xc,2)
        push!(vtk_points, MeshCell(VTKCellTypes.VTK_VERTEX, [i]))
    end
    file = vtk_grid("$(filename)_points", xc, vtk_points) do vtk
    end

    return nothing
end