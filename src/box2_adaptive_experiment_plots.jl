using JLD2, Plots, LaTeXStrings

include("resampling.jl")
include("smoothing_test.jl")
include("smoothing_test_analysis.jl")

include("box2_adaptive_experiment_constants.jl")
include("plotting_styles.jl")

out = load("$(@__DIR__)/out/box2_adaptive_experiment_summaries.jld2")

s_rel_adaptive = out["s_rel_adaptive"]
s_filt_adaptive = out["s_filt_adaptive"]

do_individual_plots = false # Whether to produce same plots as in the non-adaptive experient, for all thresholds

function triplot(p1, p2, p3)
    plot(p1, p2, p3, layout=(1,3), size=(800,300), left_margin=4Plots.mm, bottom_margin=4Plots.mm)
end

function do_normalising_constant_plots(s_rel, experiment_base)

    p1 = show_norm_const_est_styles(Δs, s_rel[:,:,1], styles; res=AllResamplings)
    plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\log(\mathrm{RMSE})")
    p2 = show_norm_const_est_styles(Δs, s_rel[:,:,1], styles; res=AllResamplings, yaxis=:none, legend=:none)
    plot!(p2, xlims=(-12.1,-3.9), ylims=(0.35,0.6), xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}")
    p = twinplot(p1, p2)
    savefig(p, "$(experiment_base)_normalising_constant_64.pdf")

    p1 = show_norm_const_est_styles(Δs, s_rel[:,:,end], styles; res=AllResamplings)
    plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\log(\mathrm{RMSE})")
    p2 = show_norm_const_est_styles(Δs, s_rel[:,:,end], styles; res=AllResamplings, yaxis=:none, ylabel="Std.", legend=:none)
    plot!(p2, xlims=(-12.1,-3.9), ylims=(0.1,0.4), xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}")
    p = twinplot(p1, p2)
    savefig(p, "$(experiment_base)_normalising_constant_512.pdf")

    ss = diagm(sqrt.(Ns))*transpose(s_rel[end,:,:])
    p1 = show_norm_const_est_styles(Ns, ss, styles; res=AllResamplings,
    xlabel=L"\log_2 N", yaxis=:none, ylabel=L"\sqrt{N} \cdot \mathrm{RMSE}")
    plot!(p1, legend=:topleft, title=L"\Delta = 2^{-12}")

    ss = diagm(sqrt.(Ns))*transpose(s_rel[4,:,:])
    p2 = show_norm_const_est_styles(Ns, ss, styles; res=AllResamplings,
    xlabel=L"\log_2 N", yaxis=:none, ylabel=L"\sqrt{N} \cdot \mathrm{RMSE}")
    plot!(p2, legend=:none, title=L"\Delta = 2^{-6}")

    p = twinplot(p1, p2)
    savefig(p, "$(experiment_base)_normalising_constant_varying_N.pdf")
end

function do_filtering_smoothing_plots(s_filt, experiment_base)
    p1 = show_norm_const_est_styles(Δs, s_filt[:,:,6,end]', styles; res=AllResamplings, ylabel="log(RMSE)")
    plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\log(\mathrm{RMSE})")
    p2 = show_norm_const_est_styles(Δs, s_filt[:,:,6,end]', styles; res=AllResamplings, yaxis=:none)
    plot!(p2, legend=:none, xlims=(-12.1, -3.9), ylims=(0.07, 0.19), xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}")
    p = twinplot(p1, p2)

    savefig(p, "$(experiment_base)_filtering_512.pdf")

    p1 = show_norm_const_est_styles(Δs, s_filt[:,:,1,end]', styles; res=AllResamplings, ylabel="log(RMSE)")
    plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\log(\mathrm{RMSE})")
    p2 = show_norm_const_est_styles(Δs, s_filt[:,:,1,end]', styles; res=AllResamplings, yaxis=:none)
    plot!(p2, legend=:none, xlims=(-12.1, -3.9), ylims=(0.15, 0.4), xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}")
    p = twinplot(p1, p2) 
    savefig(p, "$(experiment_base)_smoothing_512.pdf")


    p1 = show_norm_const_est_styles(Δs, s_filt[:,:,6,end]', styles; res=AllResamplings, ylabel="log(RMSE)", legend=:none, yaxis=:none)
    plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}", title=L"\mathrm{Filtering}")
    p2 = show_norm_const_est_styles(Δs, s_filt[:,:,1,end]', styles; res=AllResamplings, ylabel="log(RMSE)", yaxis=:none)
    plot!(p2, xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}", title=L"\mathrm{Smoothing}")
    p = twinplot(p1, p2) 
    savefig(p, "$(experiment_base)_filtering_smoothing_512.pdf")

    ss = diagm(sqrt.(Ns))*transpose(s_filt[:,end,end,:])
    p1 = show_norm_const_est_styles(Ns, ss, styles; res=AllResamplings,
    xlabel=L"\log_2 N", yaxis=:none, ylabel=L"\sqrt{N} \cdot \mathrm{RMSE}")

    plot!(p1, legend=:none, title=L"\mathrm{Filtering}") # \Delta = 2^{-12}"

    ss = diagm(sqrt.(Ns))*transpose(s_filt[:,end,1,:])
    p2 = show_norm_const_est_styles(Ns, ss, styles; res=AllResamplings,
    xlabel=L"\log_2 N", yaxis=:none, ylabel=L"\sqrt{N} \cdot \mathrm{RMSE}")

    plot!(p2, legend=:left, title=L"\mathrm{Smoothing}")
    p = twinplot(p1, p2)
    savefig(p, "$(experiment_base)_filtering_smoothing_varying_Ns.pdf")
end

# Do individual plots as in the non-adaptive case:
if do_individual_plots
    for (k,threshold) = enumerate(thresholds)
        experiment_base = "$(@__DIR__)/out/box2_adaptive_$(threshold)"
        do_normalising_constant_plots(s_rel_adaptive[k], experiment_base)
        do_filtering_smoothing_plots(s_filt_adaptive[k], experiment_base)
    end
end

experiment_base = "$(@__DIR__)/out/box2_adaptive_"

function filtering_smoothing_plots(th, N)
    N_ind = findfirst(N .== Ns)
    ind_ = findfirst(thresholds .== 0.5)

    s_filt = s_filt_adaptive[ind_]

    p1 = show_norm_const_est_styles(Δs, s_filt[:,:,6,N_ind]', styles; res=AllResamplings, ylabel="log(RMSE)", legend=:topleft, yaxis=:none)
    plot!(p1, xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}", title=L"\mathrm{Filtering}")
    p2 = show_norm_const_est_styles(Δs, s_filt[:,:,1,N_ind]', styles; res=AllResamplings, ylabel="log(RMSE)", yaxis=:none, legend=:none)
    plot!(p2, xlabel=L"\log_2 \Delta", ylabel=L"\mathrm{RMSE}", title=L"\mathrm{Smoothing}")
    p = twinplot(p1, p2)
end
p = filtering_smoothing_plots(0.5, 512)
savefig(p, "$(experiment_base)0.5_512_filtering_smoothing.pdf")
p = filtering_smoothing_plots(0.5, 64)
savefig(p, "$(experiment_base)0.5_64_filtering_smoothing.pdf")


function adaptive_summaries(show_resamplings, N)
    adaptive_ref_inds = [findfirst(keys(AllResamplings) .== r)
    for r = show_resamplings]
    
    N_ind = findfirst(N .== Ns)

    s_rel = hcat([s_rel_adaptive[i][end, adaptive_ref_inds, N_ind] for i=1:n_thres]...)
    
    p_const = plot(#xlabel="Resampling threshold", 
    ylabel="MSE", title="Normalising constant", legend=:none)
    for (i, res) = enumerate(show_resamplings)
        st = getfield(styles, res)
        plot!(p_const, thresholds, s_rel[i,:], label=String(res), 
        color=st.c, marker=st.marker)
        hline!(p_const, [s_rel[i,end]], color=st.c, alpha=0.5, label="")
    end

    s_filtering = hcat([s_filt_adaptive[i][adaptive_ref_inds, end, end, N_ind] for i=1:n_thres]...)

    s_smoothing = hcat([s_filt_adaptive[i][adaptive_ref_inds, end, 1, N_ind] for i=1:n_thres]...)

    p_filt = plot(xlabel="Resampling threshold", title="Filtering", legend=:none)
    for (i, res) = enumerate(show_resamplings)
        st = getfield(styles, res)
        plot!(p_filt, thresholds, s_filtering[i,:], label=String(res),
        color=st.c, marker=st.marker)
        hline!(p_filt, [s_filtering[i,end]], color=st.c, alpha=0.5, label="")
    end

    p_smth = plot(#xlabel="Resampling threshold", 
    title="Smoothing", legend=:none)
    for (i, res) = enumerate(show_resamplings)
        st = getfield(styles, res)
        plot!(p_smth, thresholds, s_smoothing[i,:], label=String(res),
        color=st.c, marker=st.marker)
        hline!(p_smth, [s_smoothing[i,end]], color=st.c, alpha=0.5, label="")
    end
    p_const, p_filt, p_smth
end

show_resamplings = ( 
:Stratified, :StratifiedPartition,
:Systematic, :SystematicPartition, 
:SSP, :SSPPartition,
:Killing,
:Multinomial, :Residual, 
)

p_const, p_filt, p_smth = adaptive_summaries(show_resamplings, 512)
plot!(p_const, ylims=(0.11,0.18))
plot!(p_filt, ylims=(0.07,0.15), legend=:topleft)
plot!(p_smth, ylims=(0.13,0.27))
p = triplot(p_const, p_filt, p_smth)
savefig(p, "$(experiment_base)512_varying_threshold.pdf")

p_const, p_filt, p_smth = adaptive_summaries(show_resamplings, 256)
plot!(p_const, ylims=(0.16,0.27))
plot!(p_filt, ylims=(0.10,0.20), legend=:topleft)
plot!(p_smth, ylims=(0.195,0.35))
p = triplot(p_const, p_filt, p_smth)
savefig(p, "$(experiment_base)256_varying_threshold.pdf")

p_const, p_filt, p_smth = adaptive_summaries(show_resamplings, 128)
plot!(p_const, ylims=(0.225,0.37))
plot!(p_filt, ylims=(0.135,0.27), legend=:topleft)
plot!(p_smth, ylims=(0.27,0.45))
p = triplot(p_const, p_filt, p_smth)
savefig(p, "$(experiment_base)128_varying_threshold.pdf")

p_const, p_filt, p_smth = adaptive_summaries(show_resamplings, 64)
plot!(p_const, ylims=(0.32,0.55))
plot!(p_filt, ylims=(0.2,0.45))
plot!(p_smth, ylims=(0.38,0.7))
p = triplot(p_const, p_filt, p_smth)
savefig(p, "$(experiment_base)64_varying_threshold.pdf")
