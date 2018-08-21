# ///////////////////
# Nonlinear equations
# ///////////////////
struct LorNLinEq end 

function (eq::LorNLinEq)(t, u, dudt)
    x, y, z = u
    @inbounds dudt[1] =  10 * (y - x)
    @inbounds dudt[2] =  28 *  x - y - x*z
    @inbounds dudt[3] = -8/3 * z + x*y
    return dudt
end


# //////////////////////////////////////////////
# FORCING FUNCTIONS FOR THE LINEARISED EQUATIONS
# Note that by default these functions add to 
# their last inputs, which is the only argument
# that gets modified
# //////////////////////////////////////////////

# sensitivity with respect to rho
dfdρ_forcing(t, u, dudt, v, dvdt) = (@inbounds dvdt[2] += u[1]; dvdt)

# no forcing does nothing (for the homogeneous equation)
zero_forcing(t, u, dudt, v, dvdt) = (dvdt)

# forcing based on the vector field (note this is a closure)
f_forcing(χ) = wrapped(t, u, dudt, v, dvdt) = (dvdt .+= χ.*dudt; dvdt)


# ////////////////////
# Linearised equations
# ////////////////////
struct LorLinEq{N, T<:NTuple{N, Base.Callable}}
    forcings::T # a tuple of functions with signature (t, u, dudt, v, dvdt)
end

# slurp arguments
LorLinEq(x::Vararg{Any, N}) where {N} = LorLinEq{N, typeof(x)}(x)

# defaults to homogeneous problem
LorLinEq() = LorLinEq(zero_forcing)


# Linearised equations
@generated function (eq::LorLinEq{N})(t, u, dudt, v, dvdt) where {N}
    quote
        # unpack
        x , y , z  = u
        x′, y′, z′ = v

        # homogeneous linear part
        @inbounds dvdt[1] =  10 * (y′ - x′)
        @inbounds dvdt[2] =  (28-z)*x′ - y′ - x*z′
        @inbounds dvdt[3] = -8/3*z′ + x*y′ + x′*y

        # add forcing (can be nothing too)
        Base.Cartesian.@nexprs $N i->eq.forcings[i](t, u, dudt, v, dvdt)

        return dvdt
    end
end