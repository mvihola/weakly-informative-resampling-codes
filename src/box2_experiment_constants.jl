include("smoothing_test.jl") # resampling_aliases

T_phys = 5
rep = 10_000
Δs = 2.0 .^ (-(0:2:12))
Ns = 2 .^ (6:9)

AllResamplings = resampling_aliases(Ns[1])

n_Δs = length(Δs)
n_res = length(AllResamplings)
n_Ns = length(Ns)

