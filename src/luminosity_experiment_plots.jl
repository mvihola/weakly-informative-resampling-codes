using JLD2, Plots, StatsPlots, LaTeXStrings

include("smoothing_test_analysis.jl")
include("plotting_styles.jl")

Ns = [8,16,32,64,128,256]
resamplings = (:Multinomial, :Residual, :Killing, 
:Stratified, :StratifiedPartition,
:Systematic, :SystematicPartition,
:SSP, :SSPPartition)

out = load("$(@__DIR__)/out/luminosity_experiment_summaries.jld2")
acc = out["acc"]
ire = out["ire"]
ρs1 = out["ρs1"]
mean_ire = [mean(x) for x in ire]


function quantile_plot!(plt, S, p=0.95; x=1:size(S)[1])
    alpha = (1-p)/2
    qs = [alpha, 0.5, 1-alpha]
    Q = mapslices(x->quantile(x, qs), S, dims=2)
    plot!(plt, x, Q[:,2], ribbon=(Q[:,2]-Q[:,1], Q[:,3]-Q[:,2]), color=:black,
    legend=:false, fillalpha=0.2)
end
function show_out(out; title="", transformed=false)
    labels = [string.(keys(θ_init))...]
    Thetas = out.Theta'
    if !transformed
        labels = replace.(labels, "log_" => "")
        Thetas = exp.(Thetas)
    end
    p_theta = corrplot(Thetas, size=(600,600), label=labels,
    title="Parameter posterior$title")
    # The volatilities:
    S = [out.X[j][i].s for i=1:length(out.X[1]), j=1:length(out.X)]
    
    p_data = plot(data.t, data.s, label="X_t", xlabel="t", 
    legend=false, title="Data (red), true state (blue) & latent posterior median, 50% and 95% credible intervals")
    vline!(data.τ, label="τ")

    quantile_plot!(p_data, S, 0.95; x=t)
    quantile_plot!(p_data, S, 0.5; x=t)
    plot(p_theta, p_data, layout=grid(2,1, heights=[0.8,0.2]), size=(1000,1000))
end

function show_latent(data, t, out)

    S = [out.X[j][i].s for i=1:length(out.X[1]), j=1:length(out.X)]

    p_data = plot(data.t, data.s, label="X_t", xlabel="t", 
    legend=false, title="Data (red), true state (blue) & latent posterior median, 50% and 95% credible intervals")
    vline!(data.τ, label="y")

    quantile_plot!(p_data, S, 0.95; x=t)
    quantile_plot!(p_data, S, 0.5; x=t)
    p_data
end

function plot_autocor(results, i=1; summary = r -> r.ρ[i,:], lab = hcat(String.(keys(res))...))
    colors = []
    for base_color = ( [1.0,0,0], [0,0,1.0], [0,1.0,0] )
        for k = 1:3
            col = (6 - 2(k-1))/6*base_color
            push!(colors, RGB(col...))
        end
    end
    push!(colors, :gray, :black, :magenta, :yellow, :cyan)
    col = hcat(colors...)

    v = hcat([summary(r) for r in results]...)
    plot(v, label=lab, color=col)
end

function plot_sorted_summary(results; summary = r -> r.acc, lab = hcat(String.(keys(res))...))
    n_res = length(results)
    acc = [summary(r) for r in results]
    ind = sortperm(acc)
    p = bar(1:n_res, acc[ind], orientation=:horizontal)
    yticks!(p, (1.0:n_res), lab[ind])
    p
end

grays = [RGB(g,g,g) for g in LinRange(0.3,1.0,length(Ns))]'

p_acc = bar(acc', orientation=:horizontal, 
  labels=["$N" for N in Ns'], legend=:bottomright,
  fillcolor=grays)
yticks!(p_acc, (1.0:length(resamplings)), [String(r) for r in resamplings])
plot!(p_acc, xlims=(0,0.25))
xlabel!(p_acc, L"\mathrm{Acceptance~rate}")

p_ire = show_norm_const_est_styles(Ns, log.(mean_ire), styles; 
  res = NamedTuple(resamplings .=> 1), yaxis=:none,
  xlabel = L"N", ylabel=L"\log(\mathrm{IRE})"
)
plot!(p_ire, legend=:topleft)
xticks!(p_ire, log2.(Ns), [string(N) for N in Ns])

p1 = twinplot(p_acc, p_ire; widths=(0.45,0.55))

savefig(p1, "luminosity_acc_ire.pdf")

p_rho = Vector{Any}(undef, length(Ns))

n_Ns = length(Ns)
for i = 1:n_Ns
    step = 2^(i-1)
    last = round(Int, 640/2^(n_Ns-i))
    p_rho[i] = show_norm_const_est_styles(0:step:last, ρs1[i][1:step:(last+1),:], styles; 
    res = NamedTuple(resamplings .=> 1), yaxis=:none,
    xtrans= x->x, ylabel="", xlabel="", legend=:none)
    ylims!(p_rho[i], (0,1))
    xticks!(p_rho[i], 0:6step:last)
    title!(p_rho[i], "\$N=$(Ns[i])\$")
end
p_acf = horizplot(p_rho[2:6]..., size=(800,200))
savefig(p_acf, "luminosity_acf.pdf")

