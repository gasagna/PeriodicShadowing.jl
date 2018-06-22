module PeriodicShadowing

using Flows
using BorderedMatrices

export build_lhs_block!,
       build_rhs_block!,
       get_shooting_points,
       build_ps_system,
       solve_ps,
       ith_span


# ////// indices of the i-th block //////
_blockrng(i::Integer, n::Integer) = ((i-1)*n+1):(i*n)


# utils
ith_span(i::Int, N::Int, T::Real) =  ((i-1)*T/N, i*T/N)

# ////// utils //////
function build_lhs_block!(Y::AbstractMatrix{U},
                          ψ,
                          span::Tuple{Real, Real},
                          y::X) where {U<:Real, X<:AbstractVector{U}}
    n = length(y)                             # state dimension
    size(Y) == (n, n) || error("wrong input") # checks
    for i = 1:n                               #
        y .= 0; y[i] = one(U)                 # reset initial conditions
        Y[:, i] = ψ(y, span)                  # propagate and store
    end                                       #
    return Y                                  # return
end

function build_rhs_block!(y::AbstractVector, ρ, span::Tuple{Real, Real})
    y .= 0     # set homogeneous initial condition
    ρ(y, span) # propagate
    return y   # return
end


function get_shooting_points(ϕ, x0::AbstractVector, T::Real, N::Int)
    mon = Monitor(x0, copy)                            # create monitor
    ϕ(copy(x0), (0, T), reset!(mon))                   # propagate
    [samples(mon)[i*div(length(samples(mon)), N)+1] for i = 0:N-1] # return points
end


function build_ps_system(ψ,
                         ρ,
                         ϕ,
                         f,
                         x0s::Vector{X}, 
                         T::Real) where {U<:Real, X<:AbstractVector{U}}
    # number of shooting stages and state dimension
    N, n = length(x0s), length(x0s[1])

    # ////// construct main diagonal block //////
    # allocate blocks
    A  = spzeros(U, (N+1)*n, (N+1)*n)
    a  = zeros(U, (N+1)*n)
    fT = zeros(U, (N+1)*n)
    f0 = zeros(U, (N+1)*n)

    # create temporaries
    Yi    = Matrix{U}(n, n)
    yi    = similar(x0s[1])
    tmp1  = similar(x0s[1])
    tmp2  = similar(x0s[1])

    # fill main block
    for i = 1:N
        # set data here
        rng_1 = _blockrng(i+1, n)
        rng_2 = _blockrng(i,   n)
        A[rng_1, rng_2] .= build_lhs_block!(Yi, ψ, ith_span(i, N, T), yi)
        a[rng_1]        .= build_rhs_block!(yi, ρ, ith_span(i, N, T))
    end

    # invert a
    a .*= -1

    # add diagonals
    A[diagind(A, 0)]   += -1;
    A[diagind(A, N*n)] +=  1;

    # period change
    for i = 1:N
        tmp1 .= x0s[i]
        ϕ(tmp1, ith_span(i, N, T))
        fT[_blockrng(i+1, n)] .= f(0.0, tmp1, tmp2)
    end

    # phase locking condition
    f0[_blockrng(1, n)] .= f(0.0, x0s[1], tmp1)

    # construct bordered matrix and vector
    B = BorderedMatrix(A, fT, f0, 0.0)
    b = BorderedVector(a, 0.0)

    return B, b
end

function solve_ps(ψ, ρ, ϕ, f, x0s::Vector{<:AbstractVector}, T::Real)
    # build system
    B, b = build_ps_system(ψ, ρ, ϕ, f, x0s, T)

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

end