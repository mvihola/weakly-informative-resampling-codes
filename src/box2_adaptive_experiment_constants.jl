include("box2_experiment_constants.jl")

thresholds = vcat(0.0, 
cumsum(2^(-3)*ones(7)),
1 - 2^(-4),
1 - 2^(-5),
1 - 2^(-6),
1 - 2^(-7),
1 - 2^(-8),
1 - 2^(-9),
1.0)

n_thres = length(thresholds)

