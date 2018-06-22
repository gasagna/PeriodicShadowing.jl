using Flows, PeriodicShadowing
using Base.Test

@testset "test lorenz problem                    " begin
    include("lorenz.jl")

    # parameters
    Δt = 0.5e-2

    # nonlinear integrator
    f       = LorNLinEq()
    ϕ       = integrator(f, Scheme(:CB4_4R3R, zeros(3)), Δt)
    __mon__ = Monitor(zeros(3), copy)

    # homogeneous integrator
    lf1 = LorLinEq(__mon__, 0, zero_forcing)
    ψ  = integrator(lf1, Scheme(:CB4_4R3R, zeros(3)), Δt)

    # non homogeneous integrator with zero forcing along f
    lf2 = LorLinEq(__mon__, 0, dfdρ_forcing)
    ρ  = integrator(lf2, Scheme(:CB4_4R3R, zeros(3)), Δt)

    # non homogeneous integrator with non zero forcing along f
    lf3 = LorLinEq(__mon__, 0, dfdρ_forcing)
    ρχ  = integrator(lf3, Scheme(:CB4_4R3R, zeros(3)), Δt)

    # quadrature function
    quadfun(t, x, dqdt) = (dqdt[1] = x[3]; dqdt)

    # non homogeneous integrator with non zero forcing along f with quadrature
    lf4 = LorLinEq(__mon__, 0, dfdρ_forcing)
    ρχquad  = integrator(lf3, nothing, quadfun, Scheme(:CB4_4R3R, zeros(3), zeros(1)), Δt)

    # shooting parameters
    T = 100
    N = 20

    # first land on the attractor
    x0 = ϕ(Float64[1, 0, 0], (0, 50))

    for rep = 1:10
        # propagate forward a little bit
        ϕ(x0, (0, T))
        
        # store solution in monitor
        ϕ(copy(x0), (0, T), reset!(__mon__))

        # get shooting points
        x0s = get_shooting_points(ϕ, x0, T, N)

        # solve ps system
        y0s, χ = solve_ps(ψ, ρ, ϕ, f, x0s, T)
        
        # set χ constant for non homogeneous integrators
        lf3.χ = χ
        lf4.χ = χ
        
        # periodicity check
        yend = ρχ(copy(y0s[end]), ((N-1)*T/N, T))
        δ = norm(yend - y0s[1])/norm(yend)
        
        # calculate integral
        Jρ = quadrature(ρχquad, y0s, T)

        # test against expected values
        @test abs(χ  + 0.0241)  < 1e-3
        @test abs(Jρ - 1.0170)  < 1e-2
        @test δ                 < 1e-2
    end
end