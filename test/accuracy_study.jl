module Study

using StaticArrays: SVector, @SVector
using RegionTrees
using LinearAlgebra
using PyPlot
using Random 
#using BenchmarkTools
using SparseArrays
using CloudSBP
using Krylov
using LinearOperators
using ILUZero
#using LinearOperators
#using AlgebraicMultigrid
#using IncompleteLU
#using ILUZero

Random.seed!(42)

degree = 2
Dim = 2

#num_x = [10; 20; 40; 80; 160]
num_x = [5; 10; 20; 40]

h = 1.0./num_x
L2_err = zeros(size(num_x))
proj_err = zero(L2_err)

for (k, n) in enumerate(num_x)
    println("computing error for h = ",h[k])

    println("Defining root")
    @time root = Cell(SVector(ntuple(i -> 0.0, Dim)),
                      SVector(ntuple(i -> 1.0, Dim)),
                      CellData(Vector{Int}(), Vector{Int}()))


    # xc = rand(Dim, n^Dim)

    # Quasi-uniform points
    xc = zeros(Dim, n^Dim)
    xnd = reshape(xc, (Dim, ntuple(i -> n, Dim)...))
    dx = 1/n
    for I in CartesianIndices(xnd)
        # I[1] is the coordinate, so I[I[1] + 1] is the index for that coord
        xnd[I] = (I[I[1]+1] - 1)/n + 0.5*dx
        xnd[I] += 0.25*(2*rand()-1)*dx
    end


    println("timing refine_on_points...")
    @time CloudSBP.refine_on_points!(root, xc)
    for cell in allleaves(root)
        split!(cell, CloudSBP.get_data)
    end
    println("Number of cells = ",CloudSBP.num_leaves(root))
    println("Number of DOFs  = ",size(xc,2))
    println("Ratio           = ",CloudSBP.num_leaves(root)/size(xc,2))
    println("timing build_nn_stencils...")
    @time CloudSBP.build_nn_stencils!(root, xc, degree)

    CloudSBP.set_xref_and_dx!(root, xc)
    m = CloudSBP.calc_moments!(root, degree)

    num_nodes = size(xc,2)
    dist_ref = ones(num_nodes)
    mu = 0.1
    max_rank = 50 # was 50
    H_tol = ones(num_nodes)
    vol = 1.0
    H_tol .*= 0.1*vol/num_nodes #  0.5e-5
    xc_init = deepcopy(xc)
    H = CloudSBP.opt_norm!(root, xc, degree, H_tol, mu, dist_ref, max_rank)
   
    println("!!!!!!!!!!!!!!!!  Reset NNs")
    @time CloudSBP.build_nn_stencils!(root, xc, degree)

    println("timing build_faces...")
    @time faces = CloudSBP.build_faces(root)
    println("timing build_boundary_faces...")
    @time bnd_faces = CloudSBP.build_boundary_faces(root)

    println("timing mass_matrix...")
    #CloudSBP.set_xref_and_dx!(root, xc) # <-- this has a slight impact on accuracy
    @time M = CloudSBP.mass_matrix(root, xc, degree)
    println("Number of non-zeros in M = ",nnz(M))
    #println("cond(M) = ",cond(Matrix(M)))
    #H = CloudSBP.diag_mass(root, xc, degree)
    #println("min(H) ",minimum(H))

    #H = CloudSBP.calc_cell_quad_weights(root, xc, degree)

    if false
        # preconditioner trials 
        # root_lo = Cell(SVector(ntuple(i -> 0.0, Dim)), 
        #             SVector(ntuple(i -> 1.0, Dim)),
        #             CloudSBP.CellData(Vector{Int}(), Vector{Face{2,Float64}}()))
        # CloudSBP.refine_on_points!(root_lo, xc)
        # for cell in allleaves(root_lo)
        #     split!(cell, CloudSBP.get_data)
        # end
        # CloudSBP.build_nn_stencils!(root_lo, xc, 0)
        # M_lo = CloudSBP.mass_matrix(root_lo, xc, 0)
        #P = inv(diagm(sqrt.(diag(M))))
        #println("cond(M_lo\\M) = ",cond(P*Matrix(M)*P))
        #P = inv(diagm(diag(M_lo)))
        #P = inv(diagm(diag(M)))
        #println("cond(M_lo\\M) = ",cond(P*Matrix(M)))

        #E = eigvals(P*Matrix(M)*P)
        #E = eigvals(P*Matrix(M))
        #println("large eigs = ",E[1:5])
        #println("small eigs = ",E[end-4:end])
        #fig = PyPlot.figure()
        #PyPlot.plot(E, "ko")
        #PyPlot.yscale("log")

        #Minv = CloudSBP.inv_mass_matrix(root, xc, degree)
        #Minv = diagm(1.0./diag(M))
        #H = diagm(1.0./sqrt.(diag(M)))
        #H = diagm(1.0./diag(M_lo))
        #println("norm(M) = ",norm(M, Inf))
        #println("norm(I - M) = ",norm(I - M, Inf))
        #println("norm(I - H*M*H) = ",norm(I - H*M*H, Inf))
        #P = I + M + M*M + M*M*M 
        #ml = ruge_stuben(M)
        #ml = smoothed_aggregation(M)
        #P = aspreconditioner(ml)
        #LU = ilu(M, τ = 0.0)
        #@show nnz(LU) / nnz(M)
        #fact = ilu0(M)
        #P = LinearOperator(Float64, n^Dim, n^Dim, false, false, (y, v) -> 
        #                   ldiv!(y, fact, v))

        # how many iterations does it take to solve with M?
        b = randn(size(xc,2))
        b ./= norm(b)
        println("timing CG solve...")
        @time dx, stats = cg(M, b, atol=1e-12, rtol=1.0e-10, 
                             history=true, M=I)
        #println(ml)
        #println("timing multigrid solve...")
        #@time AlgebraicMultigrid._solve(ml, b, verbose=true, maxiter=1000) # should return ones(1000)


        println(stats)
    end

    # root_ho = Cell(SVector(ntuple(i -> 0.0, Dim)), 
    #                 SVector(ntuple(i -> 1.0, Dim)),
    #                 CloudSBP.CellData(Vector{Int}(), Vector{Face{2,Float64}}()))
    # CloudSBP.refine_on_points!(root_ho, xc)
    # for cell in allleaves(root_ho)
    #     split!(cell, CloudSBP.get_data)
    # end
    # CloudSBP.build_nn_stencils!(root_ho, xc, 2*degree)
    # # #H = CloudSBP.diag_mass(root_ho, xc, 2*degree)
    # H = CloudSBP.diag_mass_ver2(root_ho, xc, 2*degree)
    # P = CloudSBP.prolongation(xc, root, xc, degree)
    # M = P'*diagm(H)*P 
    # # println("minimum(H) = ", minimum(H))
    # #H = CloudSBP.diag_mass_ver3(root, xc, degree)
    # println("minimum(H) = ", minimum(H))



    println("timing build_first_deriv...")
    #@time S = CloudSBP.build_first_deriv(root, faces, xc, degree)
    levset(x) = 1.0
#    @time S = CloudSBP.build_first_deriv(root, faces, xc, levset, degree)
    @time S = CloudSBP.build_first_deriv(root, faces, xc, degree)
    println("timing build_boundary_operator...")
    @time bnd_pts, bnd_nrm, bnd_dof, bnd_prj =
        CloudSBP.build_boundary_operator(root, bnd_faces, xc, degree)

    sbp = CloudSBP.SBP(S, bnd_pts, bnd_nrm, bnd_dof, bnd_prj)

    #println("min(H) ",minimum(H))
    #println("cond(M) = ",cond(Array(M)))

    u = zeros(size(xc,2))
    #u = exp.(xc[1,:])
    #u = ones(size(u))
    uexact = exp.(xc[1,:])
    dudx = exp.(xc[1,:])
    #CloudSBP.weak_differentiate!(dudx, u, 1, sbp)
    #dudx ./= H
    #dudx = M\dudx

    #H = CloudSBP.calc_SBP_norm(sbp, xc, degree, Dim)
    #H = CloudSBP.calc_SBP_quad(xc, 2*degree, Dim)
    #println("min(H) ",minimum(H))    
    #P = CloudSBP.prolongation(xc, root, xc, degree)

    Q = sbp.S[1] - sbp.S[1]'
    #b = P'*diagm(H)*P*dudx
    b = M*dudx 
    #b = H.*dudx
    #H = vec(sum(M,dims=1))
    #H = diag(Matrix(M))
    #b = H.*dudx
    #b = H.*dudx 
    #H = diagm(vec(sum(M,dims=1)))
    #F = ldlt(M, perm=1:size(M,1))
    #H = diag(F)
    #b = H*dudx
    for (f, Pface) in enumerate(sbp.bnd_prj)
        nrm = sbp.bnd_nrm[f]
        xf = sbp.bnd_pts[f]
        for i = 1:size(Pface,2)
            row = sbp.bnd_dof[f][i]
            for j = 1:size(Pface,2)
                col = sbp.bnd_dof[f][j]
                for q = 1:size(Pface,1)
                    if nrm[1,q] < -1e-10
                        # This is inflow 
                        #b[row] -= Pface[q,i]*nrm[1,q]*exp.(xf[1,q])
                        Q[row, col] -= 0.5*Pface[q,i]*nrm[1,q]*Pface[q,j]
                    else 
                        # This is outflow 
                        Q[row, col] += 0.5*Pface[q,i]*nrm[1,q]*Pface[q,j]
                    end
                end
            end
            for q = 1:size(Pface,1)
                if nrm[1,q] < -1e-10
                    # This is inflow 
                    b[row] -= Pface[q,i]*nrm[1,q]*exp.(xf[1,q])
                end 
            end
        end
    end

    # check that operator is exact for polynomials 
    dudx = zeros(num_nodes)
    u = vec(xc[1,:])
    println("sum(H*u) = ",dot(H, u))

    CloudSBP.weak_differentiate!(dudx, u, 1, sbp)
    println("Error dudx - exact = ", norm(dudx - M*ones(num_nodes)))
    #println(dudx - H.*ones(num_nodes))
    if k == 3
        PyPlot.scatter(vec(xc[1,:]), vec(xc[2,:]), s=6,c="k")
        PyPlot.scatter(vec(xc[1,:]), vec(xc[2,:]), s=2.0.*(20 .+ log.(abs.(dudx - H.*ones(num_nodes)))),c="r")
        #PyPlot.scatter(vec(xc[1,:]), vec(xc[2,:]), s=20 .+ log.(abs.(dudx - H.*ones(num_nodes))),c="r")
    end
    CloudSBP.weak_differentiate!(dudx, u, 2, sbp)
    println("Error dudy - exact = ", norm(dudx))

    println("timing system solve...")    
    @time u = Q\b 
    
    # println("timing cholesky factorization...")
    # @time F = cholesky(M)
    # println("timing cholesky solve...")
    # @time y = F\u

    err = u - uexact
    #err = dudx 
    #L2_err[k] = sqrt(abs(dot(err,H.*err)))
    L2_err[k] = sqrt(err'*M*err)

    # compute the projection error 
    #M_lump = vec(sum(M, dims=1))
    #u_proj = M \ (M_lump.*uexact)
    #err = u_proj - uexact
    #proj_err[k] = sqrt(err'*M*err)

    #M_lump = diagm(1.0./vec(sum(M, dims=1)))
    println("timing ILUO factorization...")
    @time prec = ilu0(M)
    prec_op = LinearOperator(Float64, num_nodes, num_nodes, false, false,
                             (y, v) -> y .= prec \ v )
    println("timing Mass matrix solve...")
    @time du, stats = cg(M, u, atol=1e-12, rtol=1.0e-10, history=true, M=prec_op)
    println(stats)
    println(stats.residuals)
        
    # plot the solution
    if k == -1
        #u .= 0.5*(1 + exp(1))
        # find the maximum number of phi basis over all cells
        max_basis = 0
        for cell in allleaves(root)
            max_basis = max(max_basis, length(cell.data.points))
        end
        num_quad = 2*degree
        x1d, w1d = CloudSBP.lgl_nodes(num_quad-1)
        wq = zeros(length(w1d)^Dim)
        xq = zeros(Dim, length(wq))
        xnd = reshape(xq, (Dim, ntuple(i -> num_quad, Dim)...))
        uq = zeros(length(wq))
        und = reshape(uq, ntuple(i -> num_quad, Dim))
        phi = zeros(length(wq), max_basis)
        for cell in allleaves(root)
            CloudSBP.quadrature!(xq, wq, cell.boundary, x1d, w1d)
            CloudSBP.dgd_basis!(phi, degree, view(xc, :, cell.data.points), xq, Val(Dim))
            fill!(uq, 0.0)
            num_basis = length(cell.data.points)
            for i = 1:num_basis
                row = cell.data.points[i]
                for q = 1:length(wq)
                    uq[q] += phi[q,i]*u[row]
                end
            end
            PyPlot.contourf(xnd[1,:,:], xnd[2,:,:], und, LinRange(1,exp(1),10))
        end
        PyPlot.plot(vec(xc[1,:]), vec(xc[2,:]), "ro")
    end

end

println("h = ",h)
println("L2_err = ", L2_err)
println("rates =  ", log.(L2_err[2:end]./L2_err[1:end-1])./log.(h[2:end]./h[1:end-1]))

#println("proj err = ", proj_err)
#println("proj err rates =  ", log.(proj_err[2:end]./proj_err[1:end-1])./log.(h[2:end]./h[1:end-1]))

# mass matrix, degree = 2
# L2_err = [0.0035789007463621378, 0.0009857855501513258, 0.00015126955826943507, 1.937283008623771e-5]
# rates =  [1.8601707950999955, 2.7041521473609533, 2.9650150554528016]

# diagonal norm, degree = 2
# L2_err = [0.1586836532290902, 0.1335949666050822, 0.02370165375177543, 0.009246636959994293]
# rates =  [0.24828786373961853, 2.494806023190637, 1.3579870727868524]

# diagonal norm SBP, degree = 2
# L2_err = [0.01984880137485628, 0.015191784132682427, 0.0006098939097776689, 0.00010093704117333771]
# rates =  [0.3857605780384469, 4.63858919093273, 2.595102607841409]

#------------------------------------------
# Quasi uniform grid

# mass matrix, degree = 2
# L2_err = [0.0029563810019985677, 0.0005090682541296983, 5.8546354891798286e-5, 1.1742426549380727e-5]
# rates =  [2.537901202009909, 3.1202078447416017, 2.3178487816774918]

# diagonal norm SBP, degree = 2
# L2_err = [0.007404612272186305, 0.003202845034248171, 0.0007795971851557617, 0.00010433789756742127]
# rates =  [1.2090701968710456, 2.0385532100942743, 2.9014656134187424]



end # Module Study