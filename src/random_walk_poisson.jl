using ComponentArrays

# Define the particle type for the model (here, latent is univariate AR(1))
mutable struct RWParticle
    s::Float64
    RWParticle() = new(0.0) # Void constructor required!
end
RWDefaultParam() = ComponentVector(σ=1.0, σ_s1=1.0, s_min=0.0, s_max=3.0)
PPDefaultParam() = ComponentVector(α=1.0, βI_0=1.0)

struct RWPPScratch{Vdt,Vobs,MT,GT}
    dt::Vdt
    obs::Vobs
    M_par::MT
    G_par::GT
end
function DefaultRWPPScratch(; τ=zeros(0), Δ=nothing, T_start=nothing, T_end=nothing)
    if isnothing(T_start)
        T_start = (length(τ)>0) ? minimum(τ) : 0.0
    end
    if isnothing(T_end)
        T_end = (length(τ)>0) ? maximum(τ) : 1.0
    end
    @assert (T_end > T_start && isfinite(T_end - T_start))
    if isnothing(Δ)
        Δ = T_end - T_start
    end
    mesh = create_mesh(τ, Δ; T_start=T_start, T_end=T_end)
    RWPPScratch(mesh.dt, mesh.obs, RWDefaultParam(), PPDefaultParam())
end

function M_reflected_RW!(x, rng, k, x_prev, scratch)
    if k == 1
        x.s = scratch.M_par.σ_s1 * randn(rng)
    else
        x.s = x_prev.s + scratch.M_par.σ * sqrt(scratch.dt[k-1]) * randn(rng)
    end
    # Reflection:
    x.s = reflect(x.s, scratch.M_par.s_min, scratch.M_par.s_max)
    nothing
end

# A 'reflection' of x onto [a,b]
function reflect(x, a, b)
    if a <= x <= b
        return x
    else
        ba = b-a
        y = abs(x - a)
        d, r = divrem(y, ba)
        if rem(d, 2) == 1
            r = ba-r
        end
        return a + r
    end
end


function lG_PP(k, x, scratch)
    if k > length(scratch.dt)
        return 0.0
    end
    @inbounds L = -scratch.dt[k] * scratch.G_par.βI_0 * exp(-scratch.G_par.α*x.s)
    @inbounds if scratch.obs[k]
        L += log(scratch.G_par.βI_0) - scratch.G_par.α*x.s
    end
    L
end

# Create time-discretisation grid
function create_mesh(τ, Δ; T_start=0.0, T_end=maximum(τ))
    t = T_start:Δ:T_end    # Uniform grid
    t = sort(union(t, τ))  # Add time of observations
    if t[end] ∈ τ
        push!(t, t[end])
    end
    dt = diff(t)
    obs = zeros(Bool, length(t))
    for k = eachindex(t)
        if t[k] ∈ τ
            obs[k] = true
        end
    end
    (dt = dt, obs=obs, t=t)
end

function set_param!(scratch, θ)
    scratch.M_par.σ = exp(θ.log_σ)
    scratch.G_par.α = exp(θ.log_α)
    scratch.G_par.βI_0 = exp(θ.log_βI_0)
end

