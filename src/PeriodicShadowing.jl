module PeriodicShadowing

using VectorPairs
using BorderedMatrices

import Flows

export get_shooting_points,
       solve_ps_plc_tan,
       solve_ps_opt_tan,
       ith_span,
       quad_χ,
       DualForcing,
       quadrature

# ////// UTILS //////

# Get `N` points along the trajectory originating at `x0`, every `T/N` time units.
function get_shooting_points(ϕ, x0::AbstractVector, T::Real, N::Int)
    mon = Flows.Monitor(x0, copy)                            # create monitor
    ϕ(copy(x0), (0, T), Flows.reset!(mon))                         # propagate
    [Flows.samples(mon)[i*div(length(Flows.samples(mon)), N)+1] for i = 0:N-1] # return points
end

# global row indices of the i-th block
_blockrng(i::Integer, n::Integer) = ((i-1)*n+1):(i*n)

# the i-th time span over the shooting domain
ith_span(i::Int, N::Int, T::Real) =  ((i-1)*T/N, i*T/N)

# quadrature function
quad_χ(t, x::VectorPair, dqdt) =
    (dqdt[1] = dot(x.v1, x.v2); dqdt[2] = dot(x.v2, x.v2); dqdt)

# used to calculate the quadratures for the calculation of χᵒ
function DualForcing(f1, f2)
    function wrapper(t, u, v::VectorPair, dvdt::VectorPair)
        f1(t, u, v.v1, dvdt.v1)
        f2(t, u, v.v2, dvdt.v2)
        return dvdt
    end
    return wrapper
end

function quadrature(ρquad)
    function wrapped!(y0s::AbstractVector, q0::AbstractVector, T::Real)
        N = length(y0s)
        for i = 1:N
            ρquad(y0s[i], q0, ith_span(i, N, T))
        end
        return q0./T
    end
end

# ////// BUILDING BLOCKS //////
#
#  Construct the principal matrix solution `Y` over time span `span`, from
#  the inital condition `x`.
#
# Arguments
# ---------
# Y    : output - will write into the columns of this matrix
# ψ    : input  - the homogeneous propagator for the linear equations
# span : input  - integrate the equations over this time span
# x    : input  - the initial state of the nonlinear equations
# y    : input  - temporary
# xcp  : input  - temporary
function _build_lhs_block!(Y::AbstractMatrix{U},
                          ψ,
                          span::Tuple{Real, Real},
                          x::X,
                          y::X,
                          xcp::X) where {U<:Real, X<:AbstractVector{U}}
    n = length(y)                                   # state dimension
    size(Y) == (n, n) || error("wrong input")       # checks
    for i = 1:n                                     #
        y .= 0; y[i] = one(U)                       # reset initial conditions
        xcp .= x                                    #
        Y[:, i] = last(ψ(Flows.couple(x, y), span)) # propagate and store
    end                                             #
    return Y                                        # return
end

#  Construct the particular solution of the inhomogeneous equations,
#  over time span `span`, from the inital condition `x`.
#
# Arguments
# ---------
# ρ    : input - the inhomogeneous propagator for the linear equations
# span : input - integrate the equations over this time span
# x    : input - the initial state of the nonlinear equations
# y    : input - temporary
# xcp  : input - temporary
function _build_rhs_block!(ρ,
                           span::Tuple{Real, Real},
                           x::X,
                           y::X,
                           xcp::X) where {X <: AbstractVector}
    y   .= 0                      # set initial condition
    xcp .= x                      #
    ρ(Flows.couple(xcp, y), span) # propagate
    return y                      # return
end

#  Construct the lhs of the multiple shooting system
#
# Arguments
# ---------
# ψ    : input - the homogeneous propagator for the linear equations
# x0s  : input - a vector of initial conditions
# T    : input - total time span
#
# Returns
# -------
# B    : the left hand side of the shooting system
function _build_lhs_all(ψ,
                        x0s::Vector{X},
                        T::Real) where {U<:Real, X<:AbstractVector{U}}
    # number of shooting stages and state dimension
    N, n = length(x0s), length(x0s[1])

    # allocate sparse matrix
    M = spzeros(U, (N+1)*n, (N+1)*n)

    # create temporaries
    Yi    = Matrix{U}(n, n)
    tmp1  = similar(x0s[1])
    tmp2  = similar(x0s[1])

    # fill main block
    for i = 1:N
        rng_1 = _blockrng(i+1, n)
        rng_2 = _blockrng(i,   n)
        M[rng_1, rng_2] .= _build_lhs_block!(Yi,
                                             ψ,
                                             ith_span(i, N, T),
                                             x0s[i],
                                             tmp1,
                                             tmp2)
    end

    # add diagonals
    M[diagind(M, 0)]   += -1
    M[diagind(M, N*n)] +=  1

    return M
end


#  Construct the rhs of the multiple shooting system
#
# Arguments
# ---------
# ρ    : input - the inhomogeneous propagator for the linear equations
# x0s  : input - a vector of initial conditions
# T    : input - total time span
#
# Returns
# -------
# b    : the right hand side of the shooting system
function _build_rhs_all(ρ,
                        x0s::Vector{X},
                        T::Real) where {U<:Real, X<:AbstractVector{U}}
    # number of shooting stages and state dimension
    N, n = length(x0s), length(x0s[1])

    # allocate
    b  = zeros(U, (N+1)*n)

    # create temporary
    tmp1 = similar(x0s[1])
    tmp2 = similar(x0s[1])

    # fill main block with negative
    for i = 1:N
        b[_blockrng(i+1, n)] .-= _build_rhs_block!(ρ,
                                                  ith_span(i, N, T),
                                                  x0s[i], tmp1, tmp2)
    end

    return b
end

# ////// SOLVERS //////

#  Construct the multiple shooting system for the tangent approach, using
#  the the orthogonality constraint at the initial time to select the
#  period gradient Tp/T.
#
# Arguments
# ---------
# ψ   : input - linear homogeneous propagator
# ρ   : input - linear inhomogeneous propagator
# ϕ   : input - nonlinear propagator
# f   : input - compute right hand side of nonliner equations
# x0s : input - vector of initial conditions at the shooting intervals
# T   : input - total trajectory length
#
# Returns
# -------
# B  : the left hand side of the shooting system
# b  : the right hand side of the shooting system
function _build_system_plc(ψ,
                           ρ,
                           ϕ,
                           f,
                           x0s::Vector{X},
                           T::Real) where {U<:Real, X<:AbstractVector{U}}
    # number of shooting stages and state dimension
    N, n = length(x0s), length(x0s[1])

    # ////// construct main diagonal block //////
    fT = zeros(U, (N+1)*n)
    f0 = zeros(U, (N+1)*n)

    # create temporaries
    tmp1  = similar(x0s[1])
    tmp2  = similar(x0s[1])

    # period change
    for i = 1:N
        tmp1 .= x0s[i]
        ϕ(tmp1, ith_span(i, N, T))
        fT[_blockrng(i+1, n)] .= f(0.0, tmp1, tmp2)
    end

    # phase locking condition
    f0[_blockrng(1, n)] .= f(0.0, x0s[1], tmp1)

    # construct bordered matrix and vector
    B = BorderedMatrix(build_ms_matrix(ψ, x0s, T), fT, f0, 0.0)
    b = BorderedVector(build_ms_vector(ρ, x0s, T), 0.0)

    return B, b
end

#  Solve the Periodic Shadowing tangent problem, using the orthogonality
#  constraint at the initial time to select the gradient Tp/T.
#
# Arguments
# ---------
# ψ   : input - linear homogeneous propagator
# ρ   : input - linear inhomogeneous propagator
# ϕ   : input - nonlinear propagator
# f   : input - compute right hand side of nonliner equations
# x0s : input - vector of initial conditions at the shooting intervals
# T   : input - total trajectory length
#
# Returns
# -------
# y0s  : a vector containing the initial conditions of the solution at the
#        beginning of the shooting intervals
# Tp/T : the period gradient
function solve_ps_plc_tan(ψ, ρ, ϕ, f, x0s::Vector{<:AbstractVector}, T::Real)
    # build system
    B, b = build_system_plc(ψ, ρ, ϕ, f, x0s, T)

    # solve out of place
    A_ldiv_B!(B, b, :BEM, false)

    # number of shooting stages and state dimension
    N, n = length(x0s), length(x0s[1])

    # unpack solution and return
    y0s = map(1:N) do i
            tmp = similar(x0s[1])
            tmp .= b[_blockrng(i, n)]
          end

    y0s, b[end]*N/T
end

#  Solve the Periodic Shadowing tangent problem, using the an optimality
#  condition to select the gradient Tp/T.
#
# Arguments
# ---------
# TODO
#
# Returns
# -------
# y0s  : a vector containing the initial conditions of the solution at the
#        beginning of the shooting intervals
# Tp/T : the period gradient
function solve_ps_opt_tan(ψ,
                          ρ_p,
                          ρ_f,
                          ρ_quad,
                          x0s::Vector{X},
                          T::Real) where {X<:AbstractVector}
    # get sizes
    N, n = length(x0s), length(x0s[1])

    # build ms matrix
    A = _build_lhs_all(ψ, x0s, T)

    # factorise ms matrix
    luA = lufact(A)

    # obtain two right hand sides
    a_p = _build_rhs_all(ρ_p, x0s, T)
    a_f = _build_rhs_all(ρ_f, x0s, T)

    # solve two problems
    _y0s_p = A_ldiv_B!(similar(a_p), luA, a_p)
    _y0s_f = A_ldiv_B!(        a_p,  luA, a_f)

    # and unpack into a vector of elements of type X
    y0s_p = map(1:N) do i
              X(_y0s_p[_blockrng(i, n)])
            end

    y0s_f = map(1:N) do i
              X(_y0s_f[_blockrng(i, n)])
            end

    # compute quadratures and the ratio
    out = ρ_quad(VectorPair.(y0s_p, y0s_f), zeros(2), T)
    χ_opt = -out[1]/out[2]

    # now compute actual solution
    y0s = map(1:N) do i
        X(_y0s_p[_blockrng(i, n)] .+ χ_opt.*_y0s_f[_blockrng(i, n)])
    end

    # return solution
    return y0s, χ_opt
end

end