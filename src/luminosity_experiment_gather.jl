using JLD2, AdaptiveParticleMCMC, Plots, Statistics, LinearAlgebra, LaTeXStrings, StatsBase

import MonteCarloMarkovKernels: estimateBM

include("random_walk_poisson.jl")
include("smoothing_test.jl") # resampling_aliases
include("smoothing_test_analysis.jl") # show_...

Ns = [8,16,32,64,128,256]
resamplings = (:Multinomial, :Residual, :Killing, 
:Stratified, :StratifiedPartition,
:Systematic, :SystematicPartition,
:SSP, :SSPPartition)

function load_luminosity(res, N; prefix="$(@__DIR__)/out/luminosity500k")
    load("$(prefix)_$(res)_$N.jld2")["result"]
end

result = Matrix{Any}(undef, length(Ns), length(resamplings))
Ns = sort(Ns; rev=true)
for (i,N) in enumerate(Ns)
    for (j,r) in enumerate(resamplings)
        result[i,j] = load_luminosity(String(r), N)
    end
end

# Acceptance rates
acc = [r.acc for r in result]

# Calculate 'ground truth' mean over everything
μs = [mapslices(mean, r.Theta, dims=2) for r in result]
μ_ground = mean(μs)

# Make chains zero-mean
centred = [r.Theta - repeat(μ_ground, 1, size(r.Theta,2)) for r in result]

# Calculate second moment estimates
μ2s = [mapslices(x->mean(x.^2), e, dims=2) for e in centred]
# ...and 'ground truth' std
σ_ground = sqrt.(mean(μ2s))

# Standardise errors
z = [e./σ_ground for e in centred]

# Asymptotic variance estimates
asvar = [mapslices(estimateBM, e, dims=2) for e in z]

# Inverse relative efficiencies
ire = diagm(Ns)*asvar

# Autocorrelations
ρs = [mapslices(x->autocor(x, 0:640; demean=false), e, dims=2) for e in z]

# First component
ρs1 = [hcat([r[1,:] for r in ρs[i,:]]...) for i in 1:size(ρs,1)]

@save "$(@__DIR__)/out/luminosity_experiment_summaries.jld2" acc ire ρs1