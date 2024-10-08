# Tests related to building the first-derivative operator 

"""
    y = skew_matvec(S, x)

Multiplies the sparse matrix `S` by `x`, assuming that `S` stores the upper part
of a skew-symmetric matrix.
"""
function skew_matvec(S, x)
    y = zero(x)
    rows = rowvals(S)
    vals = nonzeros(S)
    m, n = size(S)
    for j = 1:n
        for i in nzrange(S, j)
            row = rows[i]
            val = vals[i]
            y[row] += val*x[j]
            y[j] -= val*x[row]
        end
    end
    return y 
end






# @testset "test uncut_volume_integrate!: dimension $Dim" for Dim in 1:3

#     # use a unit HyperRectangle 
#     root = Cell(SVector(ntuple(i -> 0.0, Dim)),
#                 SVector(ntuple(i -> 1.0, Dim)),
#                 CellData(Vector{Int}(), Vector{Int}()))

#     # storage for sparse matrix information 
#     rows = Array{Array{Int64}}(undef, Dim)
#     cols = Array{Array{Int64}}(undef, Dim)
#     Svals = Array{Array{Float64}}(undef, Dim)
#     for d = 1:Dim
#         rows[d] = Int[]
#         cols[d] = Int[]
#         Svals[d] = Float64[]
#     end

#     # for each polynomial degree `deg` from 1 to max_degree, we form all
#     # possible monomials u and v such that their total degree is `deg` or less.
#     # These monomials are evaluated at the random nodes `points`.  Then we
#     # contract these arrays with the skew symmetric SBP operator in each
#     # direction, and check for accuracy against the expected integral over a
#     # unit HyperRectangle.
#     max_degree = 4
#     for deg = 1:max_degree
#         num_basis = binomial(Dim + deg, Dim)
#         points = randn(Dim, num_basis) .+ 0.5*ones(Dim)
#         root.data.points = 1:num_basis        
#         CloudSBP.uncut_volume_integrate!(rows, cols, Svals, root, points, deg)
#         S = SVector(ntuple(d -> sparse(rows[d], cols[d], Svals[d]), Dim))

#         for Iu in CartesianIndices(ntuple(i -> 0:deg, Dim))
#             # Iu[1] is power for x, Iu[2] is power for y, ...
#             if sum([Iu[i] for i = 1:Dim]) <= deg 
#                 # u = (x.^Iu[1]).*(y.^Iu[2]).* ...
#                 u = vec(prod(points.^Tuple(Iu), dims=1))
#                 for Iv in CartesianIndices(ntuple(i -> 0:deg, Dim))
#                     # Iv[1] is power for x, Iv[2] is power for y, ...
#                     if sum([Iv[i] for i = 1:Dim]) <= deg
#                         # v = (x.^Iv[1]).*(y.^Iv[2]).* ...
#                         v = vec(prod(points.^Tuple(Iv), dims=1))
#                         pow_sum = MVector(ntuple(i -> Iu[i] + Iv[i] + 1, Dim))
#                         for d = 1:Dim 
#                             vSu = dot(v, skew_matvec(S[d], u))
#                             pow_sum[d] -= 1 
#                             if pow_sum[d] == 0 
#                                 # in this case, Iu[d] + Iv[d] = 0
#                                 integral = 0.0 
#                             else 
#                                 integral = 0.5*(Iu[d] - Iv[d])/prod(pow_sum)
#                             end 
#                             if abs(integral) < 1e-13
#                                 @test isapprox(vSu, integral, atol=1e-10)
#                             else
#                                 @test isapprox(vSu, integral)
#                             end
#                             pow_sum[d] += 1
#                         end
#                     end
#                 end
#             end
#         end
#         # clear memory from rows, cols, Svals 
#         for d = 1:Dim
#             resize!(rows[d], 0)
#             resize!(cols[d], 0)
#             resize!(Svals[d], 0) 
#         end
#     end
# end 

# @testset "test cut_volume_integrate!: dimension $Dim" for Dim in 2:3

#     # use a unit HyperRectangle 
#     root = Cell(SVector(ntuple(i -> 0.0, Dim)),
#                 SVector(ntuple(i -> i == 1 ? 2.0 : 1.0, Dim)),
#                 CellData(Vector{Int}(), Vector{Int}()))

#     # define a level-set that cuts the HyperRectangle in half 
#     num_basis = 1
#     xc = 0.5*ones(Dim, num_basis)
#     xc[1, 1] = 1.0
#     nrm = zeros(Dim, num_basis)
#     nrm[1, 1] = 1.0
#     tang = zeros(Dim, Dim-1, num_basis)
#     tang[:, :, 1] = nullspace(reshape(nrm[:, 1], 1, Dim))
#     crv = zeros(Dim-1, num_basis)
#     rho = 100.0*num_basis    
#     levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

#     # # define a level-set that cuts the HyperRectangle into a Simplex 
#     # num_basis = Dim
#     # xc = zeros(Dim, num_basis)
#     # nrm = zeros(Dim, num_basis)
#     # tang = zeros(Dim, Dim-1, num_basis)
#     # crv = zeros(Dim-1, num_basis)
#     # for d = 1:Dim 
#     #     idx = d
#     #     xc[d, idx] = 1.0
#     #     nrm[:, idx] = ones(Dim)/sqrt(Dim)
#     #     tang[:, :, idx] = nullspace(reshape(nrm[:,idx], 1, Dim))
#     # end
#     # rho = 100.0*num_basis    
#     # levset = LevelSet{Dim,Float64}(xc, nrm, tang, crv, rho)

#     # storage for sparse matrix information 
#     rows = Array{Array{Int64}}(undef, Dim)
#     cols = Array{Array{Int64}}(undef, Dim)
#     Svals = Array{Array{Float64}}(undef, Dim)
#     for d = 1:Dim
#         rows[d] = Int[] 
#         cols[d] = Int[] 
#         Svals[d] = Float64[]
#     end

#     # for each polynomial degree `deg` from 1 to max_degree, we form all
#     # possible monomials u and v such that their total degree is `deg` or less.
#     # These monomials are evaluated at the random nodes `points`.  Then we
#     # contract these arrays with the skew symmetric SBP operator in each
#     # direction, and check for accuracy against the expected integral over a
#     # unit HyperRectangle.
#     max_degree = 1
#     for deg = 1:max_degree
#         num_basis = binomial(Dim + deg, Dim)
#         points = randn(Dim, num_basis) .+ 0.5*ones(Dim)
#         root.data.points = 1:num_basis        
#         CloudSBP.cut_volume_integrate!(rows, cols, Svals, root, levset, points, deg)
#         S = SVector(ntuple(d -> sparse(rows[d], cols[d], Svals[d]), Dim))

#         for Iu in CartesianIndices(ntuple(i -> 0:deg, Dim))
#             # Iu[1] is power for x, Iu[2] is power for y, ...
#             if sum([Iu[i] for i = 1:Dim]) <= deg 
#                 # u = (x.^Iu[1]).*(y.^Iu[2]).* ...
#                 u = vec(prod(points.^Tuple(Iu), dims=1))
#                 for Iv in CartesianIndices(ntuple(i -> 0:deg, Dim))
#                     # Iv[1] is power for x, Iv[2] is power for y, ...
#                     if sum([Iv[i] for i = 1:Dim]) <= deg
#                         # v = (x.^Iv[1]).*(y.^Iv[2]).* ...
#                         v = vec(prod(points.^Tuple(Iv), dims=1))
#                         pow_sum = MVector(ntuple(i -> Iu[i] + Iv[i] + 1, Dim))
#                         for d = 1:Dim 
#                             vSu = dot(v, skew_matvec(S[d], u))
#                             pow_sum[d] -= 1 
#                             if pow_sum[d] == 0 
#                                 # in this case, Iu[d] + Iv[d] = 0
#                                 integral = 0.0 
#                             else 
#                                 integral = 0.5*(Iu[d] - Iv[d])/prod(pow_sum)
#                             end 
#                             if abs(integral) < 1e-13
#                                 @test isapprox(vSu, integral, atol=1e-10)
#                             else
#                                 @test isapprox(vSu, integral)
#                             end
#                             pow_sum[d] += 1
#                         end
#                     end
#                 end
#             end
#         end
#         # clear memory from rows, cols, Svals 
#         for d = 1:Dim
#             resize!(rows[d], 0)
#             resize!(cols[d], 0)
#             resize!(Svals[d], 0) 
#         end
#     end
# end 