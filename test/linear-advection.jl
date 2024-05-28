module LinearAdvection

using CutDGD
using Test
using RegionTrees
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using LevelSets
using CxxWrap
using SparseArrays
using CutQuad
using Sobol

# Following StaticArrays approach of using repeatable "random" tests
Random.seed!(42)

Dim = 2
vel = ones(Dim)

uexact(x::AbstractVector) = exp(sum(x))
dudx(x::AbstractVector) = Dim*exp(sum(x))
uexact(x::AbstractMatrix) = exp.(sum(x,dims=1))'
dudx(x::AbstractMatrix) = Dim*exp.(sum(x,dims=1))'

levset = x -> 1.0 #norm(x .- SVector(ntuple(i -> 0.5, Dim)))^2 - 0.25^2
function levset_grad!(g, x)
    g[:] = zeros(Dim) #2.0*x .- SVector(ntuple(i -> 1.0, Dim))
    return nothing
end

# all boundaries have characteristic BCs applied to them
bc_map = Dict{Any,String}("ib" => "upwind")
for di = 1:Dim 
    bc_map[di*2-1] = "upwind"
    bc_map[di*2] = "upwind"
end

L2err = zeros(2, 4)

for (dindex, degree) in enumerate(1:2)

    num_basis = binomial(Dim + 2*degree-1, Dim)

    for (nindex, num_nodes) in enumerate([10, 20, 40, 80].^Dim) #   [10, 20, 40].*num_basis) #, 80].*num_basis)

        # use a unit HyperRectangle 
        root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                    SVector(ntuple(i -> 1.0, Dim)),
                    CellData(Vector{Int}(), Vector{Int}()))

        # xc = rand(Dim, num_nodes)
        # for i in axes(xc,2)
        #     while levset(view(xc,:,i)) < 0.0
        #         xc[:,i] = rand(Dim)
        #     end
        # end
        
        xc = zeros(Dim, num_nodes)
        ptr = 1
        for x in skip(SobolSeq(Dim),num_nodes)
            if levset(x) < 0
                continue
            end
            xc[:,ptr] = x
            if ptr == num_nodes
                break
            end
            ptr += 1
        end

        # refine mesh, build stencil, get face lists
        CutDGD.refine_on_points!(root, xc)
        for cell in allleaves(root)
            split!(cell, CutDGD.get_data)
        end
        CutDGD.mark_cut_cells!(root, levset)
        CutDGD.build_nn_stencils!(root, xc, 2*degree-1)
        CutDGD.set_xref_and_dx!(root, xc)
        ifaces = CutDGD.build_faces(root)
        bfaces = CutDGD.build_boundary_faces(root)
        CutDGD.mark_cut_faces!(ifaces, levset)
        CutDGD.mark_cut_faces!(bfaces, levset)


        m = CutDGD.calc_moments!(root, levset, 2*degree-1, min(degree,2))
        dist_ref = ones(num_nodes)
        mu = 0.1
        max_rank = min(50, num_nodes) # was 50
        H_tol = ones(num_nodes)
        vol = 1.0
        H_tol .*= 0.1*vol/num_nodes #  0.5e-5
        xc_init = deepcopy(xc)
        H = CutDGD.opt_norm!(root, xc, 2*degree-1, H_tol, mu, dist_ref, 
                             max_rank)
        println("minimum(H) = ",minimum(H))

        sbp = CutDGD.build_first_derivative(root, bc_map, ifaces, bfaces, xc, 
                                            levset, levset_grad!, degree,
                                            fit_degree=min(degree,2))

        # Use the SBP operator to define the linear system
        A = similar(sbp.S[1] + sbp.S[1]')
        fill!(A, zero(Float64))
        b = zeros(num_nodes)

        # first apply the skew-symmetric part of the operator
        for di = 1:Dim 
            A += vel[di]*(sbp.S[di] - sbp.S[di]')
        end

        # now loop over the boundary
        for (bc_type, bndry) in sbp.E
            if bc_type == "upwind"
                for (xq, nrm, dof, prj) in zip(bndry.xq_face, bndry.nrm_face, 
                                               bndry.dof_face, bndry.prj_face)
                    for i in axes(prj,2)
                        row = dof[i]
                        for j in axes(prj,2)
                            col = dof[j]
                            for q in axes(prj,1)
                                velnrm = dot(vel, nrm[:,q])
                                if velnrm < 0.0
                                    # This is inflow 
                                    A[row,col] -= 0.5*prj[q,i]*velnrm*prj[q,j]
                                else 
                                    # This is outflow 
                                    A[row,col] += 0.5*prj[q,i]*velnrm*prj[q,j]
                                end
                            end
                        end
                        for q in axes(prj,1)
                            velnrm = dot(vel, nrm[:,q])
                            if velnrm < 0.0
                                # This is inflow
                                b[row] -= prj[q,i]*velnrm*uexact(xq[:,q])
                            end
                        end
                    end
                end
            end
        end

        # finally, add the MMS terms
        b += sbp.H .* dudx(xc)

        # solve and compute error 
        println("size(A) = ",size(A))
        println("rank(A) = ",rank(A))
        #B = Matrix(A)
        #println("B = ",B)
        #println("b = ",b)
        println("minimum(abs(H)) = ",minimum(abs.(sbp.H)))
        println("minimum(H) = ",minimum(sbp.H))
        #error("STOP")
        #println("cond(A) = ",cond(B))
        u = A\b 
        #u = B\b 
        du = u - uexact(xc)
        L2err[dindex, nindex] = sqrt(dot(du,sbp.H.*du))
        println("degree = ",degree,": num_nodes = ",num_nodes,": error = ",
                L2err[dindex, nindex])

    end # num_nodes loop
end # degree loop


end # module LinearAdvection