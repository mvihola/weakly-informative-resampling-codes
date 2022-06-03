using JLD2, Plots, LaTeXStrings

include("resampling.jl")
include("smoothing_test.jl")
include("smoothing_test_analysis.jl")

include("box2_adaptive_experiment_constants.jl")

L = [zeros(n_res, n_Δs, rep, n_Ns) for _ in thresholds]
X = [zeros(n_res, n_Δs, rep, T_phys+1, n_Ns) for _ in thresholds]

for (k,threshold) = enumerate(thresholds)
    for (i,N) = enumerate(Ns)
        experiment_base = "$(@__DIR__)/out/box2_adaptive_$(threshold)"
        L[k][:,:,:,i], X[k][:,:,:,:,i] = gather_experiment("$(experiment_base)", N, AllResamplings, Δs, T_phys, rep)
    end
end

L_ground = [log_mean(L[k][:,j,:,:]) for j = 1:n_Δs, k = 1:n_thres]
L_ground_truth = mapslices(log_mean, L_ground, dims=2)

s_rel_adaptive = [norm_const_relative_std_pooled(L[i]; global_mean=true, means=L_ground_truth) for i = 1:n_thres]

s_filt_adaptive = [sqrt.(smoothing_mse_pooled(X[i], L[i])) for i = 1:n_thres]

@save "$(@__DIR__)/out/box2_adaptive_experiment_summaries.jld2" s_rel_adaptive s_filt_adaptive
