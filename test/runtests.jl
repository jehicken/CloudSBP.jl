using CutDGD
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using LevelSets
using CxxWrap
using SparseArrays

# Following StaticArrays approach of using repeatable "random" tests
Random.seed!(42)

# Need tests for the following files
# * mass.jl 
# * first_derivative.jl 


@testset "CutDGD.jl" begin
    
    @testset "test DGD basis routines" begin 
        include("test_dgd_basis.jl")
    end


    if false
    @testset "test utility routines" begin 
        include("test_utils.jl")
    end

    @testset "test orthogonal polynomials" begin 
        include("test_orthopoly.jl")
    end

    @testset "test face-related routines" begin 
        include("test_face.jl")
    end

    @testset "test mesh routines" begin 
        include("test_mesh.jl")
    end

    @testset "test quadrature routines" begin 
        include("test_quadrature.jl")
    end

    @testset "test DGD basis routines" begin 
        include("test_dgd_basis.jl")
    end

    @testset "test norm routines" begin
        include("test_norm.jl")
    end

    @testset "test first-derivative routines" begin 
        include("test_first_derivative.jl")
    end

    end

end
