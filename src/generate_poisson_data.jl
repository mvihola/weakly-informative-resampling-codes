using Random, Distributions

include("random_walk_poisson.jl")
include("resampling.jl")

function generate_rw_trajectory(scr)
    T = length(scr.dt)+1
    X = [RWParticle() for i=1:T]
    rng = Random.GLOBAL_RNG
    
    x_prev = nothing
    for k = 1:T
        x = X[k]
        M_reflected_RW!(x, rng, k, x_prev, scr)
        x_prev = x
    end
    [x.s for x in X], vcat(0.0, cumsum(scr.dt))
end

function generate_poisson_observations(scr, s)
    T = length(scr.dt)
    t = vcat(0.0, cumsum(scr.dt))

    λs = scr.G_par.βI_0 *  scr.dt .* exp.( -scr.G_par.α*s[1:T] )
    λ_tot = sum(λs)

    # Total number of observations
    n = rand(Poisson(λ_tot))
    # Draw random samples
    p = λs/λ_tot
    res = MultinomialResampling(n)
    ind = zeros(Int, n)
    resample!(res, ind, p)

    τ = zeros(n)
    for (k, i) = enumerate(ind)
        τ[k] = t[i] + scr.dt[i]*rand()
    end
    sort!(τ)

    τ
end

function generate_rw_pp_scenario(scr)
    s, t = generate_rw_trajectory(scr)
    τ = generate_poisson_observations(scr, s)
    (s=s, t=t, τ=τ)
end
