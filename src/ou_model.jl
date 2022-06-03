# Ornstein-Uhlenbeck model

# Define the particle type for the model (here, latent is univariate AR(1))
mutable struct OUParticle
    s::Float64
    OUParticle() = new(0.0) # Void constructor required!
end
Base.isless(a::OUParticle, b::OUParticle) = isless(a.s, b.s)

# Model parameters:
mutable struct OUParam
    Δ::Float64     # Discretisation step size
    σ_OU::Float64     # Latent OU noise sd
    θ::Float64     # OU drift rate
    ρ::Float64     # AR(1) param
    σ::Float64     # AR(1) noise
    σ_s1::Float64  # AR(1) stationary sd
end
function ar1_from_ou(Δ, σ_OU, θ)
    ρ = exp(-θ*Δ)
    σ = sqrt( 0.5*σ_OU^2/θ * (1.0 - exp(-2*θ*Δ)) )
    σ_s1 = σ/sqrt(1.0 - ρ^2)
    ρ, σ, σ_s1
end

function OUParam(Δ=0.5, σ_OU=1.0, θ=0.1) 
    ρ, σ, σ_s1 = ar1_from_ou(Δ, σ_OU, θ)
    OUParam(Δ, σ_OU, θ, ρ, σ, σ_s1)
end

# Transition *simulator* of stationary zero-mean AR(1) with
# parameters (ρ, σ)
function M_ar1!(x, rng, k, x_prev, scratch)
    if k == 1
        x.s = scratch.par.σ_s1 * randn(rng)
    else
        x.s = scratch.par.ρ * x_prev.s + scratch.par.σ * randn(rng)
    end
    nothing
end

function set_delta!(par, Δ)
    ρ, σ, σ_s1 = ar1_from_ou(Δ, par.σ_OU, par.θ)
    par.Δ = Δ
    par.ρ = ρ
    par.σ = σ
    par.σ_s1 = σ_s1
    nothing
end
