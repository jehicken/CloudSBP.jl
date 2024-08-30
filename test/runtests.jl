using CloudSBP
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using SparseArrays
using CutQuad

# For repeatable "random" tests
Random.seed!(42)

@testset "CloudSBP.jl" begin

    if true
        
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

    @testset "test node optimization routines" begin
        include("test_node_opt.jl")
    end

    @testset "test symmetric-part routines" begin
        include("test_symmetric_part.jl")
    end

    @testset "test skew-part routines" begin
        include("test_skew_part.jl")
    end

    @testset "test first-derivative routines" begin 
        include("test_first_derivative.jl")
    end

    @testset "test dissipation routines" begin 
        include("test_dissipation.jl")
    end

    end

end
