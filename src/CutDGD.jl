module CutDGD

using LinearAlgebra
using SparseArrays
using RegionTrees
import RegionTrees: AbstractRefinery, needs_refinement, refine_data
using StaticArrays: SVector, @SVector
using NearestNeighbors
import SpecialFunctions: gamma
using LevelSets
using CutQuad
using CxxWrap

export CellData, Face

"""
    get_complex_step(T)
returns an appropriate complex-step size for the given type
"""
get_complex_step(::Type{T}) where {T <: Float32} = 1f-20
get_complex_step(::Type{T}) where {T <: Float64} = 1e-60
get_complex_step(::Type{T}) where {T <: ComplexF32} = 1f-20
get_complex_step(::Type{T}) where {T <: ComplexF64} = 1e-60

# Module variable used to store a reference to the levset;
# this is needed for the @safe_cfunction macro
const mod_levset = Ref{Any}()

include("utils.jl")
include("orthopoly.jl")
include("face.jl")
include("mesh.jl")
include("quadrature.jl")
include("dgd_basis.jl")
include("mass.jl")
include("norm.jl")
include("symmetric_part.jl")
include("skew_part.jl")
include("first_derivative.jl")
include("dissipation.jl")


function output_solution(root::Cell{Data, Dim, T, L}, xc, degree, u;    
                         num_pts=degree+1) where {Data, Dim, T, L}

    num_cell = 0
    for cell in allleaves(root)
        if !isempty(cell.data.points)
            num_cell += 1
        end
    end
    xplot = Vector{Matrix{T}}(undef, num_cell)
    uplot = Vector{Vector{T}}(undef, num_cell)

    ptr = 1
    for cell in allleaves(root)
        if isempty(cell.data.points) continue end
        # create the plot mesh for this cell 
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

end
