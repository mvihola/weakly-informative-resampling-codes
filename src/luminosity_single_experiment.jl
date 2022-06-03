using JLD2, AdaptiveParticleMCMC, LaTeXStrings, Random

Random.seed!(12345)

include("random_walk_poisson.jl")
include("smoothing_test.jl") # resampling_aliases
include("smoothing_test_analysis.jl") # horizplot
include("luminosity_experiment_plots.jl")

import AdaptiveParticleMCMC: _run_smc!, _run_csmc!,
_log_likelihood, _set_model_param!, _pick_particle!,
_init_path_storage, _copy_reference!,
_save_reference!, _reference_log_likelihood
include("gen_smc.jl")

@load "luminosity_data.jld2" # -> data

# "Physical time" of the scenario
N = 32
iter = 500_000

RWPPScratch() = DefaultRWPPScratch(; τ=data.τ, Δ=0.1, T_start=0.0, T_end=200.0)
T = length(RWPPScratch().dt)

resampling = Symbol(:SystematicPartition)
res = resampling_aliases(N)[resampling]

# Normal prior:
prior(θ) = -.2*mapreduce(t->t^2, +, θ)

θ_init = ComponentVector(log_α=0.0, log_σ=0.0, log_βI_0=0.0)

gstate = GenSMCState(T, N, RWParticle, RWPPScratch, M_reflected_RW!, lG_PP;
 set_param! = set_param!, resampling = res )

out = adaptive_pmmh(θ_init, prior, gstate, iter; save_paths = true, thin = 10)

t = cumsum(RWPPScratch().dt)

p_latent = show_latent(data, t, out)

plot!(p_latent, title="", ylabel=L"X_t", xlabel=L"t")
p = horizplot(p_latent; size=(800,200))
savefig(p, "luminosity_data_result.pdf")
