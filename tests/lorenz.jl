using Flows, VectorPairs

# Nonlinear equations
struct LorNLinEq 
    ρ::Float64
end 

function (eq::LorNLinEq)(t::Real, u::AbstractVector, dudt::AbstractVector)
    x, y, z = u
    dudt[1] =   10 * (y - x)
    dudt[2] =   eq.ρ *  x - y - x*z
    dudt[3] = -8/3 * z + x*y
    return dudt
end

# Linearised equations
mutable struct LorLinEq{M<:Flows.AbstractMonitor, F}
        mon::M
          χ::Float64
    forcing::F
          u::Vector{Float64}
       dudt::Vector{Float64}
end

LorLinEq(mon::M, 
         χ::Real=0, 
         forcing::Base.Callable=zero_forcing) where {M<:Flows.AbstractMonitor} =
    LorLinEq{M, typeof(forcing)}(mon, χ, forcing, zeros(3), zeros(3))


# Linearised equations
function (eq::LorLinEq)(t::Real, v::AbstractVector, dvdt::AbstractVector)
    x′, y′, z′ = v

    # interpolate solution
    eq.mon(eq.u,    t, Val{0}())
    eq.mon(eq.dudt, t, Val{1}())

    # extract components
    x, y, z = eq.u

    # homogeneous linear part is the default
    dvdt[1] =  10 * (y′ - x′)
    dvdt[2] =  (28-z)*x′ - y′ - x*z′
    dvdt[3] = -8/3*z′ + x*y′ + x′*y

    # add forcing (can be nothing)
    eq.forcing(t, eq.u, eq.dudt, v, dvdt)

    # add χ⋅dudt id needed
    if eq.χ != 0
        dvdt[1] += eq.χ * (   10 * (y - x) )
        dvdt[2] += eq.χ * (   28 * x - y - x*z )
        dvdt[3] += eq.χ * ( -8/3 * z + x*y )
    end
    
    return dvdt
end

# sensitivity with respect to rho
dfdρ_forcing(t, u, dudt, v, dvdt) = (@inbounds dvdt[2] += u[1]; dvdt)

# no forcing (for the homogeneous equation)
zero_forcing(t, u, dudt, v, dvdt) = (dvdt)

# forcing dxdt
f_forcing(t, u, dudt, v, dvdt) = (dvdt .+= dudt; dvdt)