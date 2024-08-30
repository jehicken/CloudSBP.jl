
"""
    y = apply_dissipation(diss, x)

Computes `(Rl - Rr)^T W (Rl - Rr)x` using the given dissipation operator and 
returns the result.
"""
function apply_dissipation(diss, x)
    y = zero(x)
    z = (diss.R_left - diss.R_right)*x 
    z .*= diss.w_face
    y = (diss.R_left' - diss.R_right')*z
    return y 
end

# passes for Dim = 3, but takes 6 minutes
@testset "test build_dissipation: dimension $Dim, degree $degree" for Dim in 1:2, degree in 1:4

    @testset "uncut domain" begin

        # use a unit HyperRectangle 
        root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                    SVector(ntuple(i -> 1.0, Dim)),
                    CellData(Vector{Int}(), Vector{Int}()))

        num_basis = binomial(Dim + 2*degree-1, Dim)
        num_nodes = 5*num_basis
        xc = rand(Dim, num_nodes)

        # refine mesh, build stencil, get face lists
        CloudSBP.refine_on_points!(root, xc)
        for cell in allleaves(root)
            split!(cell, CloudSBP.get_data)
        end
        levset = x -> 1.0
        CloudSBP.build_nn_stencils!(root, xc, 2*degree-1)
        CloudSBP.set_xref_and_dx!(root, xc)
        ifaces = CloudSBP.build_faces(root)

        # construct the dissipation operator 
        diss = CloudSBP.build_dissipation(ifaces, xc, degree, levset)

        # check that polynomials of `degree` are in the null space of diss
        num_basis = binomial(Dim + degree, Dim)
        V = zeros(num_nodes, num_basis)

        for i in axes(V,2)
            for di = 1:Dim
                y = apply_dissipation(diss, view(V, :, i))
                @test isapprox(norm(y), 0.0, atol=10^degree*1e-12)
            end
        end 
    end

    @testset "cut domain" begin

        # use a unit HyperRectangle 
        root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                    SVector(ntuple(i -> 1.0, Dim)),
                    CellData(Vector{Int}(), Vector{Int}()))

        levset = x -> norm(x .- SVector(ntuple(i -> 0.5, Dim)))^2 - 0.25^2
        #levset = x -> 0.5 - x[1]

        num_basis = binomial(Dim + 2*degree-1, Dim)
        num_nodes = 5*num_basis
        xc = rand(Dim, num_nodes)

        for i in axes(xc,2)
            while levset(view(xc,:,i)) < 0.0
                xc[:,i] = rand(Dim)
            end
        end

        # refine mesh, build stencil, get face lists
        CloudSBP.refine_on_points!(root, xc)
        for cell in allleaves(root)
            split!(cell, CloudSBP.get_data)
        end
        CloudSBP.mark_cut_cells!(root, levset)
        CloudSBP.build_nn_stencils!(root, xc, 2*degree-1)
        CloudSBP.set_xref_and_dx!(root, xc)
        ifaces = CloudSBP.build_faces(root)
        CloudSBP.mark_cut_faces!(ifaces, levset)

        # construct dissipation operator 
        diss = CloudSBP.build_dissipation(ifaces, xc, degree, levset, 
                                        fit_degree=min(degree, 2))

        # check that polynomials of `degree` are in the null space of diss
        num_basis = binomial(Dim + degree, Dim)
        V = zeros(num_nodes, num_basis)

        for i in axes(V,2)
            for di = 1:Dim
                y = apply_dissipation(diss, view(V, :, i))
                @test isapprox(norm(y), 0.0, atol=10^degree*1e-12)
            end
        end 
    end
end