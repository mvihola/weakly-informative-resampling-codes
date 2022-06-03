using JLD2, Plots, LaTeXStrings

include("resampling.jl")
include("smoothing_test.jl")
include("smoothing_test_analysis.jl")

include("box2_experiment_constants.jl")
include("plotting_styles.jl")

out = load("$(@__DIR__)/out/box2_experiment_summaries.jld2")
s_rel = out["s_rel"]
s_filt = out["s_filt"]

p1 = show_norm_const_est_styles(Δs, s_rel[:,:,1], styles; res=AllResamplings)
plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\log(\mathrm{RMSE})")
p2 = show_norm_const_est_styles(Δs, s_rel[:,:,1], styles; res=AllResamplings, yaxis=:none, legend=:none)
plot!(p2, xlims=(-12.1,-3.9), ylims=(0.35,0.6), xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}")
p = twinplot(p1, p2)
savefig(p, "box2_normalising_constant_64.pdf")

p1 = show_norm_const_est_styles(Δs, s_rel[:,:,end], styles; res=AllResamplings)
plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\log(\mathrm{RMSE})")
p2 = show_norm_const_est_styles(Δs, s_rel[:,:,end], styles; res=AllResamplings, yaxis=:none, ylabel="Std.", legend=:none)
plot!(p2, xlims=(-12.1,-3.9), ylims=(0.1,0.4), xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}")
p = twinplot(p1, p2)
savefig(p, "box2_normalising_constant_512.pdf")

st = (; filter(p->first(p) ∉ (:Multinomial, :Residual), collect(pairs(styles)))...)

ss = diagm(sqrt.(Ns))*transpose(s_rel[end,:,:])
p1 = show_norm_const_est_styles(Ns, ss, st; res=AllResamplings,
xlabel=L"\log_2 N", yaxis=:none, ylabel=L"\sqrt{N} \cdot \mathrm{RMSE}")
plot!(p1, legend=:topleft, title=L"\Delta = 2^{-12}")

ss = diagm(sqrt.(Ns))*transpose(s_rel[4,:,:])
p2 = show_norm_const_est_styles(Ns, ss, st; res=AllResamplings,
xlabel=L"\log_2 N", yaxis=:none, ylabel=L"\sqrt{N} \cdot \mathrm{RMSE}")
plot!(p2, legend=:none, title=L"\Delta = 2^{-6}")

p = twinplot(p1, p2)
savefig(p, "box2_normalising_constant_varying_N.pdf")


p1 = show_norm_const_est_styles(Δs, s_filt[:,:,6,end]', styles; res=AllResamplings, ylabel="log(RMSE)")
plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\log(\mathrm{RMSE})")
p2 = show_norm_const_est_styles(Δs, s_filt[:,:,6,end]', styles; res=AllResamplings, yaxis=:none)
plot!(p2, legend=:none, xlims=(-12.1, -3.9), ylims=(0.07, 0.19), xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}")
p = twinplot(p1, p2)

savefig(p, "box2_filtering_512.pdf")

p1 = show_norm_const_est_styles(Δs, s_filt[:,:,1,end]', styles; res=AllResamplings, ylabel="log(RMSE)")
plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\log(\mathrm{RMSE})")
p2 = show_norm_const_est_styles(Δs, s_filt[:,:,1,end]', styles; res=AllResamplings, yaxis=:none)
plot!(p2, legend=:none, xlims=(-12.1, -3.9), ylims=(0.15, 0.4), xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}")
p = twinplot(p1, p2) 
savefig(p, "box2_smoothing_512.pdf")


p1 = show_norm_const_est_styles(Δs, s_filt[:,:,6,end]', st; res=AllResamplings, ylabel="log(RMSE)", legend=:none, yaxis=:none)
plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}", title=L"\mathrm{Filtering}")
p2 = show_norm_const_est_styles(Δs, s_filt[:,:,1,end]', st; res=AllResamplings, ylabel="log(RMSE)", yaxis=:none)
plot!(p2, xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}", title=L"\mathrm{Smoothing}")
p = twinplot(p1, p2) 
savefig(p, "box2_filtering_smoothing_512.pdf")

st_ = (; filter(p->first(p) ∉ (:Multinomial, :Residual, :StratifiedPartition), collect(pairs(styles)))...)


ss = diagm(sqrt.(Ns))*transpose(s_filt[:,end,end,:])
p1 = show_norm_const_est_styles(Ns, ss, st_; res=AllResamplings,
xlabel=L"\log_2 N", yaxis=:none, ylabel=L"\sqrt{N} \cdot \mathrm{RMSE}")

plot!(p1, legend=:none, title=L"\mathrm{Filtering}") # \Delta = 2^{-12}"

ss = diagm(sqrt.(Ns))*transpose(s_filt[:,end,1,:])
p2 = show_norm_const_est_styles(Ns, ss, st_; res=AllResamplings,
xlabel=L"\log_2 N", yaxis=:none, ylabel=L"\sqrt{N} \cdot \mathrm{RMSE}")

plot!(p2, legend=:left, title=L"\mathrm{Smoothing}")

p = twinplot(p1, p2)

savefig(p, "box2_filtering_smoothing_varying_Ns.pdf")

