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
using PyPlot

# Following StaticArrays approach of using repeatable "random" tests
Random.seed!(42)

Dim = 2
vel = ones(Dim)

uexact(x::AbstractVector) = exp(sum(x))
dudx(x::AbstractVector) = Dim*exp(sum(x))
uexact(x::AbstractMatrix) = exp.(sum(x,dims=1))'
dudx(x::AbstractMatrix) = Dim*exp.(sum(x,dims=1))'

# levset = x -> 1.0 
# function levset_grad!(g, x)
#     g[:] = zeros(Dim)
#     return nothing
# end

# levset = x -> -x[1] + 0.5 - pi/1000
# function levset_grad!(g, x)
#     g[:] = [-1.0; 0.0]
#     return nothing
# end

radius = 0.25 #+ pi/1000
levset = x -> norm(x .- SVector(ntuple(i -> 0.5, Dim)))^2 - radius^2
function levset_grad!(g, x)
    g[:] = 2.0*x .- SVector(ntuple(i -> 1.0, Dim))
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
    #for (nindex, numx) in enumerate([10, 20, 40, 80])

        # use a unit HyperRectangle 
        root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                    SVector(ntuple(i -> 1.0, Dim)),
                    CellData(Vector{Int}(), Vector{Int}()))

        # root = Cell(SVector(ntuple(i -> 0.0, Dim)),
        #             SVector(ntuple(i -> i == 1 ? 0.5 : 1.0, Dim)),
        #             CellData(Vector{Int}(), Vector{Int}()))

        # xc = rand(Dim, num_nodes)
        # for i in axes(xc,2)
        #     while levset(view(xc,:,i)) < 0.0
        #         xc[:,i] = rand(Dim)
        #     end
        # end
                        
        xc = zeros(Dim, num_nodes)
        ptr = 1
        for x in skip(SobolSeq(Dim), 2*num_nodes)
            if levset(x) < 0
                continue
            end
            xc[:,ptr] = x
            if ptr == num_nodes
                break
            end
            ptr += 1
        end
        #xc[1,:] .*= 0.5 # !!!!!!!!!!!!!!!!!!

        # # Quasi-uniform points
        # xcart = zeros(Dim, numx^Dim)
        # xnd = reshape(xcart, (Dim, ntuple(i -> numx, Dim)...))
        # dx = 1/numx
        # for I in CartesianIndices(xnd)
        #     # I[1] is the coordinate, so I[I[1] + 1] is the index for that coord
        #     xnd[I] = (I[I[1]+1] - 1)/numx + 0.5*dx
        #     xnd[I] += 0.25*(2*rand()-1)*dx
        # end
        # num_nodes = 0
        # xc = zeros(Dim,0)
        # for i in axes(xcart,2)
        #     if levset(xcart[:,i]) > 0
        #         # lots of allocations here
        #         xc = hcat(xc, xcart[:,i])
        #         num_nodes += 1
        #     end
        # end

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

        for cell in allleaves(root)
            if CutDGD.is_immersed(cell)
                continue
            end
            if length(cell.data.points) <= 0
                println("Found cell with no points assigned?")
            end
        end

        #m = CutDGD.calc_moments!(root, levset, 2*degree-1, min(degree,2))
        m = CutDGD.calc_moments!(root, levset, max(2,2*degree-1), 2)
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
                                            fit_degree=2) #min(degree,2))

        diss = CutDGD.build_dissipation(ifaces, xc, degree, levset, 
                                        fit_degree=2) #min(degree, 2))

        # Use the SBP operator to define the linear system
        A = similar(sbp.S[1] + sbp.S[1]')
        fill!(A, zero(Float64))
        b = zeros(num_nodes)

        # first apply the skew-symmetric part of the operator
        for di = 1:Dim 
            A += vel[di]*(sbp.S[di] - sbp.S[di]')
        end

        # now loop over the boundary
        check_area = 0.0
        numf = 0
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
                    # if size(xq,2) > 0 
                    #     if abs(xq[1,1] - 0.5) < 1e-10
                    #         numf += 1
                    #     end
                    # end
                    for q in axes(prj,1)
                        if norm(xq[:,q] .- 0.5)  < 0.3 
                            #println("xq = ",xq[:,q])
                            #println("nrm = ",nrm[:,q])
                            check_area += norm(nrm[:,q])
                        end
                    end

                end
            end
            println("number of faces total = ",size(bndry.xq_face))
        end

        println("number of east faces = ",numf)
        println("check area = ",check_area)

        # apply the dissipation 
        A += (diss.R_left' - diss.R_right')*spdiagm(diss.w_face)*(diss.R_left - diss.R_right)

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
        #println("cond(A) = ",cond(Matrix(A)))
        u = A\b 
        #u = B\b 
        du = u - uexact(xc)
        L2err[dindex, nindex] = sqrt(dot(du,sbp.H.*du))
        println("degree = ",degree,": num_nodes = ",num_nodes,": error = ",
                L2err[dindex, nindex])

        if dindex == 2 && nindex == 1

            # plot the domain and quadrature points 
            fig = figure("quad_points",figsize=(10,10))

            xplot, uplot = CutDGD.output_solution(root, xc, degree, du)
            vmin = minimum(du)
            vmax = maximum(du)
            for (xp, up) in zip(xplot, uplot)
                PyPlot.contourf(reshape(xp[1,:], (degree+1,degree+1)),
                                reshape(xp[2,:], (degree+1,degree+1)),
                                reshape(up, (degree+1,degree+1)), vmin=vmin,
                                vmax=vmax)
                PyPlot.contourf(reshape(xp[1,:], (degree+1,degree+1)),
                                reshape(xp[2,:], (degree+1,degree+1)),
                                reshape(up, (degree+1,degree+1)),
                                levels=LinRange(vmin, vmax, 10))
            end
            theta = LinRange(0, 2*pi, 100)
            PyPlot.plot(radius*cos.(theta) .+ 0.5, radius*sin.(theta) .+ 0.5, "k-")

            # num_quad1d = degree + 1
            # x1d, w1d = CutDGD.lg_nodes(num_quad1d) # could also use lgl_nodes
            # wq = zeros(length(w1d)^Dim)
            # xq = zeros(Dim, length(wq))
            

            # for cell in allleaves(root)
            #     if CutDGD.is_immersed(cell) 
            #         continue
            #     end
            #     v = hcat(collect(vertices(cell.boundary))...)
            #     PyPlot.plot(v[1,[1,2,4,3,1]], v[2,[1,2,4,3,1]], "-k")
            #     if CutDGD.is_cut(cell)
            #         wq_cut, xq_cut = cut_cell_quad(cell.boundary, levset, num_quad1d, 
            #         fit_degree=min(degree,2))
            #         PyPlot.plot(vec(xq_cut[1,:]), vec(xq_cut[2,:]), "gs")
                    
            #         surf_wts, surf_pts = cut_surf_quad(cell.boundary, levset, num_quad1d,
            #         fit_degree=min(degree,2))
            #         PyPlot.plot(vec(surf_pts[1,:]), vec(surf_pts[2,:]), "rd")
            #     else
            #         CutDGD.quadrature!(xq, wq, cell.boundary, x1d, w1d)
            #         PyPlot.plot(vec(xq[1,:]), vec(xq[2,:]), "bs")
            #     end
            # end
            
            # # # plot the interface quad points
            # # for face in ifaces 
            # #     if CutDGD.is_immersed(face)
            # #         continue
            # #     end
            # #     if CutDGD.is_cut(face)
            # #         wq_face, xq_face = cut_face_quad(face.boundary, face.dir, levset,
            # #         num_quad1d, fit_degree=min(degree,2))
            # #         PyPlot.plot(vec(xq_face[1,:]), vec(xq_face[2,:]), "rd")
            # #     else
            # #         wq_face = zeros(length(w1d)^(Dim-1))
            # #         xq_face = zeros(Dim, length(wq_face))
            # #         CutDGD.face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, face.dir)
            # #         PyPlot.plot(vec(xq_face[1,:]), vec(xq_face[2,:]), "bd")
            # #     end
                
            # # end

            # #check_area = 0.0
            # #numf = 0
            # for face in bfaces
            #     if CutDGD.is_immersed(face)
            #         continue
            #     end
            #     if CutDGD.is_cut(face)
            #         wq_face, xq_face = cut_face_quad(face.boundary, abs(face.dir), levset,
            #         num_quad1d, fit_degree=min(degree,2))
            #         PyPlot.plot(vec(xq_face[1,:]), vec(xq_face[2,:]), "yd", ms=10)
            #     else
            #         wq_face = zeros(length(w1d)^(Dim-1))
            #         xq_face = zeros(Dim, length(wq_face))
            #         CutDGD.face_quadrature!(xq_face, wq_face, face.boundary, x1d, w1d, abs(face.dir))
            #         PyPlot.plot(vec(xq_face[1,:]), vec(xq_face[2,:]), "bd")
            #         # if abs(xq_face[1,1] - 0.5) < 1e-10
            #         #     numf += 1
            #         # end
            #         # for q in axes(xq_face,2)
            #         #     if abs(xq_face[1,q] - 0.5) < 1e-10
            #         #         check_area += wq_face[q]
            #         #     end
            #         # end
            #     end
            # end
            # #println("number of east faces = ", numf)
            # #println("check_area during plot = ",check_area)
            
        end

    end # num_nodes loop
end # degree loop


end # module LinearAdvection