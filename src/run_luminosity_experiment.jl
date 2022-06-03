using ArgParse, JLD2, AdaptiveParticleMCMC

include("random_walk_poisson.jl")
include("smoothing_test.jl") # resampling_aliases

import AdaptiveParticleMCMC: _run_smc!, _run_csmc!,
_log_likelihood, _set_model_param!, _pick_particle!,
_init_path_storage, _copy_reference!,
_save_reference!, _reference_log_likelihood
# ... and override with this:
include("gen_smc.jl")

@load "luminosity_data.jld2" # -> data

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--resampling", "-r"
            help = "Resampling scheme"
            arg_type = String
            default = "Killing"
        "--num_particles", "-N"
            help = "Number of particles"
            arg_type = Int
            default = 128
        "--iterations", "-i"
            arg_type = Int
            default = 10_000
        "--output", "-o"
            arg_type = String
            help = "File name where output is stored"
            required = true
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

# "Physical time" of the scenario
N = parsed_args["num_particles"]
iter = parsed_args["iterations"]
out = parsed_args["output"]

RWPPScratch() = DefaultRWPPScratch(; τ=data.τ, Δ=0.1, T_start=0.0, T_end=200.0)
T = length(RWPPScratch().dt)

resampling = Symbol(parsed_args["resampling"])
res = resampling_aliases(N)[resampling]


# Normal prior:
prior(θ) = -.2*mapreduce(t->t^2, +, θ)

θ_init = ComponentVector(log_α=0.0, log_σ=0.0, log_βI_0=0.0)

gstate = GenSMCState(T, N, RWParticle, RWPPScratch, M_reflected_RW!, lG_PP;
 set_param! = set_param!, resampling = res )

result = adaptive_pmmh(θ_init, prior, gstate, iter)

jldsave(out; result)
