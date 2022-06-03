using JLD2, Plots, LaTeXStrings

include("resampling.jl")
include("smoothing_test.jl")
include("smoothing_test_analysis.jl")
include("box2_experiment_constants.jl")

L = zeros(n_res, n_Δs, rep, n_Ns)
X = zeros(n_res, n_Δs, rep, T_phys+1, n_Ns)

for (i,N) = enumerate(Ns)
    L[:,:,:,i], X[:,:,:,:,i] = gather_experiment("$(@__DIR__)/out/box2", N, AllResamplings, Δs, T_phys, rep)
end

s_rel = norm_const_relative_std_pooled(L; global_mean=true)
s_filt = sqrt.(smoothing_mse_pooled(X, L))

@save "$(@__DIR__)/out/box2_experiment_summaries.jld2" s_rel s_filt

