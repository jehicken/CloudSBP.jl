module CutDGD

using LinearAlgebra
using SparseArrays
using RegionTrees
import RegionTrees: AbstractRefinery, needs_refinement, refine_data
using StaticArrays: SVector, @SVector
using NearestNeighbors
import SpecialFunctions: gamma
using CutQuad
using WriteVTK
using Random
#using LinearOperators, Krylov
using Tulip, JuMP

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
include("norm_ext.jl")
include("symmetric_part.jl")
include("skew_part.jl")
include("first_derivative.jl")
include("dissipation.jl")
include("output.jl")



end
