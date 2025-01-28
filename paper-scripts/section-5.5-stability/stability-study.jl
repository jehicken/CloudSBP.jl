module StabilityStudy 
# This module verifies the stability of the SBP discretization and provides a 
# paraview file of the error in the solution at the final time.

using CloudSBP
using StaticArrays: SVector, @SVector, MVector
using LinearAlgebra
using Random
using SparseArrays
using Arpack
using DifferentialEquations

# For repeatable "random" tests
Random.seed!(42)

# study parameters
add_diss = false
diss_coeff = 0.25
Dim = 2
vel = ones(Dim)
deg = [1;2;3;4]
num1d = [48]
pert_size = 0.25
growth = 0.1 # 2.0 for non-uniform, 0.1 for approximately uniform
output_solution = true

tspan = [0; 2*pi]
Gamma = pi
function velocity(x)
    r = norm(x)
    utheta = Gamma/(2*pi*r)
    return Float64[-utheta*x[2]/r; utheta*x[1]/r]
end

# all boundaries have characteristic BCs applied to them
bc_map = Dict{Any,String}("ib" => "upwind")
for di = 1:Dim 
    bc_map[di*2-1] = "upwind"
    bc_map[di*2] = "upwind"
end

function uinitial(x)
    # find the center of the bump
    rc = 0.75
    c = [rc; 0.0]
    # find the distance between x and c 
    dx = norm(x - c)
    # compute the bump
    sigma = 0.5 # 0.2
    return exp(-dx*dx/sigma^2)
end

"""
    xinit = calc_xinit(x, T)    

Computes the location of a particle moving backwards in time from `x` for a duration `T`.
"""
function calc_xinit(x, T)
    function f(x, p, t)
      return -velocity(x)
    end
    tspan = [0; T]
    prob = ODEProblem(f, x, tspan)
    sol = solve(prob, Vern9(), reltol = 1e-10, abstol=1e-12)
    return sol[end]
end

"""
    calc_uexact(xc, uexact, T)

Returns (in `uexact`) the exact solution at `xc` for duration `T`.
"""
function calc_uexact(xc, uexact, T)
    @assert( size(xc,2) == length(uexact) )
    for i in axes(xc,2)
        xinit = calc_xinit(xc[:,i], T)
        uexact[i] = uinitial(xinit)
    end
    return nothing
end

function levset(x)
    r = norm(x)
    return -(r^2 - 1)*(r^2 - 0.25)
end
function levset_grad!(g, x)
    r = norm(x)
    # dphidr is divided by r to avoid issues
    dphidr = -2*(r^2 - 0.25) - (r^2 - 1)*2
    g[:] = dphidr*x
    return nothing
end

"""
    xc, num_nodes = get_points(nr, nt[, pert=0.1, growth=5.0])

Generate quasi-uniform points by perturbing a uniform grid
"""
function get_points(nr, nt; pert::Float64=0.1, growth::Float64=5.0)
    @assert(Dim == 2)
    radius(xi) = (exp(growth*xi) - 1.0)/(exp(growth) - 1.0)
    dradius(xi) = growth*exp(growth*xi)/(exp(growth)-1.0)
    rin = 0.5
    rout = 1.0
    xc = zeros(Dim, nr*nt)
    xnd = reshape(xc, (Dim, nr, nt))
    dtheta = 2*pi/nt
    H_tol = zeros(size(xc,2))
    Hnd = reshape(H_tol, (nr, nt))
    for i in axes(xnd,2) # r loop
        dr = dradius((i-1)/nr + 0.5/nr)/nr
        for j in axes(xnd,3) # theta loop
            xi = radius((i-1)/nr + 0.5/nr + pert*(2*rand()-1)/nr)
            r = (1-xi)*rin + xi*rout
            theta = (j-1)*dtheta + 0.5*dtheta + pert*dtheta*(2*rand()-1)
            xnd[1,i,j] = r*cos(theta)
            xnd[2,i,j] = r*sin(theta)
            Hnd[i,j] = 0.1*dr*dtheta*r
        end
    end
    return xc, nr*nt, H_tol
end

"""
    A, spect_est = build_system_matrix(sbp, diss, xc)

Returns a sparse matrix representation of the discretization system matrix.
"""
function build_system_matrix(sbp, diss, xc, vel)
    
    # Use the SBP operator to define the linear system
    num_nodes = length(sbp.H)
    A = similar(sbp.S[1] + sbp.S[1]')
    fill!(A, zero(Float64))
    
    # first apply the skew-symmetric part of the operator
    for di = 1:Dim 
        A += 0.5*spdiagm(vec(vel[di,:]))*(sbp.S[di] - sbp.S[di]')
        A -= 0.5*(sbp.S[di]' - sbp.S[di])*spdiagm(vec(vel[di,:]))
    end
    
    # now loop over the boundary...
    for (bc_type, bndry) in sbp.E
        if bc_type == "upwind"
            for (xq, nrm, dof, prj) in zip(bndry.xq_face, bndry.nrm_face, 
                                           bndry.dof_face, bndry.prj_face)
                for q in axes(prj,1)
                    for i in axes(prj,2)
                        row = dof[i]

                        for j in axes(prj,2)
                            col = dof[j]
                            #res[row] += 0.25*prj[q,i]*veln_u_face
                            A[row, col] += 0.25*prj[q,i]*prj[q,j]*(
                                vel[1,col]*nrm[1,q] + vel[2,col]*nrm[2,q])
                            #res[row]-= 0.25*vel[1,row]*prj[q,i]*nrm[1,q]*u_face
                            A[row, col] -= 0.25*vel[1,row]*prj[q,i]*nrm[1,q]*prj[q,j]
                            #res[row]-= 0.25*vel[2,row]*prj[q,i]*nrm[2,q]*u_face
                            A[row,col] -= 0.25*vel[2,row]*prj[q,i]*nrm[2,q]*prj[q,j]
                        end

                    end
                end
            end
        end
    end

    if add_diss
        # apply the dissipation 
        face_scale = zero(diss.w_face)
        for f in axes(diss.w_face,1)
            face_scale[f] = diss_coeff*diss.w_face[f] * abs( 
                            diss.x_face[mod(diss.dir[f],2) + 1,f])
        end
        A += (diss.R_left' - diss.R_right')*spdiagm(face_scale)*(diss.R_left - diss.R_right)
    end

    # apply the inverse norm 
    A = spdiagm(1.0./sbp.H) * A

    # estimate the spectral radius using Gershgorin's theorem
    spect_est = maximum(sum(abs.(A),dims=2))

    return A, spect_est
end

"""
    udotres = calc_residual!(res, t, u, xc, vel, sbp, diss, work)

Evaluates the residual of the linear-advection problem over the annulus and 
stores the result in `res`.  The current time is `t` and the state at the nodes 
is given by `u`.  The SBP nodes are `xc` and the velocity at the nodes is 
`vel`.  The SBP operators themselves are defined in `sbp` and the dissipation 
is defined in `diss`.  `work` is an `num_nodes x 2` array for work space.
"""
function calc_residual!(res, t, u, xc, vel, sbp::CloudSBP.SBP{T,Dim}, diss, work) where {T,Dim}
    fill!(res, 0.0)
    # first apply the skew-symmetric part of the operator
    for di = 1:Dim
        rows = rowvals(sbp.S[di])
        vals = nonzeros(sbp.S[di])
        for j in axes(sbp.S[di],2)
            velj = vel[di,j]
            for i in nzrange(sbp.S[di], j)
                row = rows[i]
                val = vals[i]
                veli = vel[di,row]
                coeff = val*0.5*(veli + velj)
                res[row] += coeff*u[j]
                res[j] -= coeff*u[row]
            end
        end
    end
    
    # now loop over the boundary...
    for (bc_type, bndry) in sbp.E
        if bc_type == "upwind"
            for (xq, nrm, dof, prj) in zip(bndry.xq_face, bndry.nrm_face, 
                                           bndry.dof_face, bndry.prj_face)
                for q in axes(prj,1)
                    # get normal velocity at quadrature point
                    veln_u_face = 0.0
                    u_face = 0.0
                    for i in axes(prj,2)
                        veln_u_face += prj[q,i]*(vel[1,dof[i]]*nrm[1,q] + 
                                            vel[2,dof[i]]*nrm[2,q])*u[dof[i]]
                        u_face += prj[q,i]*u[dof[i]]
                    end
                    for i in axes(prj,2)
                        row = dof[i]
                        res[row] += 0.25*prj[q,i]*veln_u_face
                        res[row] -= 0.25*vel[1,row]*prj[q,i]*nrm[1,q]*u_face
                        res[row] -= 0.25*vel[2,row]*prj[q,i]*nrm[2,q]*u_face
                    end
                end
            end
        end
    end

    # apply the dissipation 
    if add_diss
        du = view(work, :, 1)
        ur = view(work, :, 2)
        du[:] = diss.R_left*u
        ur[:] = diss.R_right*u
        du[:] -= ur[:]
        for f in axes(diss.dir,1)
            velface = [-diss.x_face[2,f]; diss.x_face[1,f]]
            du[f] *= diss_coeff*diss.w_face[f]*abs(velface[diss.dir[f]])
        end
        res[:] += diss.R_left'*du 
        res[:] -= diss.R_right'*du
    end

    udotres = dot(u, res)

    # apply the inverse norm 
    res[:] ./= sbp.H
    return udotres
end

"""
    udotres = step_rk4!(u, uold, t, dt, work func!)

One step of classical RK4 from `t` to `t+dt`.  Returns the product between the 
residual at `t` and the solution `uold`.
"""
function step_rk4!(u, uold, t, dt, work, func!)
    k1 = view(work, :, 1)
    k2 = view(work, :, 2)
    k3 = view(work, :, 3)
    k4 = view(work, :, 4)
    
    udotres = func!(k1, t, uold)
    u .= uold + (0.5*dt)*k1

    func!(k2, t + 0.5*dt, u)
    u .= uold + (0.5*dt)*k2

    func!(k3, t + 0.5*dt, u)
    u .= uold + dt*k3 

    func!(k4, t + dt, u)
    u .= uold + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return -udotres
end

"""
    t, sol, udotres = solve_rk4(tspan, num_steps, uinit, func! [, output=false])

Solve the unsteady problem defined by `func!` using RK4.
"""
function solve_rk4(tspan, num_steps, uinit, func!; output::Bool=false)
    dt = (tspan[2] - tspan[1])/num_steps 
    println("num_steps = ", num_steps, ": dt = ",dt)
    work = zeros(length(uinit), 4)
    udotres = zeros(num_steps)
    t = LinRange(tspan[1], tspan[2], num_steps+1)    
    u_old = deepcopy(uinit)
    u = zero(uinit)
    for k in 1:num_steps
        udotres[k] = step_rk4!(u, u_old, t[k], dt, work, func!)
        u_old[:] = u[:]
    end
    return t, u, udotres
end

origin = SVector(ntuple(i -> -1.0, Dim))
widths = SVector(ntuple(i -> 2.0, Dim))

for (i, nr) in enumerate(num1d)
    # generate mesh for this sample 
    ntheta = 6*nr
    xc, num_nodes, H_tol = get_points(nr, ntheta, pert=pert_size, growth=growth)
    vel = zero(xc)
    uinit = zeros(num_nodes)
    for i in axes(xc,2)
        vel[:,i] = velocity(xc[:,i])
        uinit[i] = uinitial(xc[:,i])
    end 

    min_widths = ones(Dim)/(20*nr)

    uexact = zeros(num_nodes)
    calc_uexact(xc, uexact, tspan[end])

    for (dindex, degree) in enumerate(deg)
        
        println()
        println(repeat("=",80))

        mesh = CloudSBP.build_mesh(xc, widths, levset, min_widths,
                                   origin=origin)

        CloudSBP.build_cell_stencils!(mesh, xc, 2*degree - 1)
        max_stencil, avg_stencil = CloudSBP.stencil_stats(mesh)
        println("degree = ",degree,": max_stencil = ",max_stencil,
                ": avg_stencil = ",avg_stencil)
        m = CloudSBP.calc_moments!(mesh.root, levset, max(2,2*degree-1), 2)

        H, success = CloudSBP.solve_norm!(mesh.root, xc, 2*degree-1, H_tol,
                                          verbose=true)
        minH = minimum(H)
        println("minimum(H) = ", minH)
        
        @assert( minH > 0.0 )
        sbp = CloudSBP.build_first_derivative(mesh, bc_map, xc, levset, 
                                              levset_grad!, degree,
                                              fit_degree=2)
        diss = CloudSBP.build_face_dissipation(mesh.ifaces, xc, degree, 
                                               levset, fit_degree=2)

        # define the ODE function `func!` for RK4
        work = zeros(size(diss.x_face,2),2)
        function func!(r, t, u)
            udotres = calc_residual!(r, t, u, xc, vel, sbp, diss, work)
            r .*= -1.0
            return udotres
        end
        
        # compute the spectral radius
        A, spect_est = build_system_matrix(sbp, diss, xc, vel)
        E, nconv, niter, nmult, resid = eigs(A)
        magE = abs.(E)
        spect = maximum(magE)
        println("spectral radius = $spect: spect. est. = $spect_est: ratio = ",spect_est/spect)
        
        dt = 2.0/spect
        println("dt = ",dt)
        num_steps = round(Int, (tspan[2] - tspan[1])/dt, RoundUp)
        println("num_steps = ",num_steps)
        
        # solve using RK4
        solve_rk4([tspan[1],tspan[1]+1e-10], 1, uinit, func!)
        @time t, ufinal, udotres = solve_rk4(tspan, num_steps, uinit, func!)

        # write the udotres history to file 
        if add_diss
            f = open("udotres_diss_p$degree.dat", "w")
        else
            f = open("udotres_nodiss_p$degree.dat", "w")
        end
        for k in axes(t, 1)
            print(f, t[k], " ")
        end
        println(f)
        for k in axes(udotres,1)
            print(f, udotres[k], " ")
        end
        # missing the last step's udotres
        res = zero(uinit)
        udotres_last = -func!(res, t[end], ufinal)
        println(f, udotres_last)
        close(f)

        if output_solution
            du = abs.(ufinal - uexact)
            if add_diss
                CloudSBP.points_vtk(xc, filename="./paraview/solution-nodes-p$degree")
                CloudSBP.output_vtk(mesh.root, xc, degree, uinit, filename="./paraview/initial-condition-p$degree")
                CloudSBP.output_vtk(mesh.root, xc, degree, ufinal, filename="./paraview/final-solution-p$degree")
                CloudSBP.output_vtk(mesh.root, xc, degree, du, filename="./paraview/final-error-p$degree")
            else 
                CloudSBP.points_vtk(xc, filename="./paraview/solution-nodes-nodiss-p$degree")
                CloudSBP.output_vtk(mesh.root, xc, degree, uinit, filename="./paraview/initial-condition-nodiss-p$degree")
                CloudSBP.output_vtk(mesh.root, xc, degree, ufinal, filename="./paraview/final-solution-nodiss-p$degree")
                CloudSBP.output_vtk(mesh.root, xc, degree, du, filename="./paraview/final-error-nodiss-p$degree")
            end
        end
    end
end

end # module