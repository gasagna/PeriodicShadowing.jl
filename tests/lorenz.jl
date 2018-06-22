using Flows

# Nonlinear equations
struct LorNLinEq end 

function (eq::LorNLinEq)(t::Real, u::AbstractVector, dudt::AbstractVector)
    x, y, z = u
    dudt[1] =   10 * (y - x)
    dudt[2] =   28 *  x - y - x*z
    dudt[3] = -8/3 * z + x*y
    return dudt
end

# Linearised equations
mutable struct LorLinEq{M<:Flows.AbstractMonitor, F}
        mon::M
          χ::Float64
    forcing::F
        tmp::Vector{Float64}
end

LorLinEq(mon::M, 
         χ::Real=0, 
         forcing::Base.Callable=zero_forcing) where {M<:Flows.AbstractMonitor} =
    LorLinEq{M, typeof(forcing)}(mon, χ, forcing, zeros(3))


# Linearised equations
function (eq::LorLinEq)(t::Real, v::AbstractVector, dvdt::AbstractVector)
    x′, y′, z′ = v

    # interpolate solution
    eq.mon(eq.tmp, t, Val{0}())
    x, y, z = eq.tmp

    # homogeneous linear part is the default
    dvdt[1] =  10 * (y′ - x′)
    dvdt[2] =  (28-z)*x′ - y′ - x*z′
    dvdt[3] = -8/3*z′ + x*y′ + x′*y

    # add forcing (can be nothing)
    eq.forcing(t, eq.tmp, v, dvdt)

    # add χ⋅dudt id needed
    if eq.χ != 0
        dvdt[1] += eq.χ * (   10 * (y - x) )
        dvdt[2] += eq.χ * (   28 * x - y - x*z )
        dvdt[3] += eq.χ * ( -8/3 * z + x*y )
    end
    
    return dvdt
end

# some sensitivity equation forcing
dfdρ_forcing(t, u, v, dvdt) = (dvdt[2] += u[1]; dvdt)
zero_forcing(t, u, v, dvdt) = (dvdt)

function quadrature(ρquad, y0s::AbstractVector, T::Real)
    q0 = zeros(1)
    y0i = similar(y0s[1])
    N = length(y0s)
    for i = 1:N
        # set initial condition
        y0i .= y0s[i]
        # integrate
        ρquad(y0i, q0, ith_span(i, N, T))
    end
    return q0[1]/T
end