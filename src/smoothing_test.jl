function single_smoothing_experiment(res, Δs, T_phys, rep, N,
    FKParticle, FKScratch, M_ar1!, lG; 
    f = x -> x.s, 
    set_delta! = (par, Δ) -> nothing,
    adaptive_threshold = nothing)
    
    n_Δs = length(Δs)
    d_f = Int(T_phys+1)
    
    L = zeros(n_Δs, rep)
    X = zeros(n_Δs, rep, d_f)
    out = zeros(d_f)
    for (j,Δ) in enumerate(Δs)
        T = 1 + Int(floor(T_phys/Δ))
        gstate = GenSMCState(T, N, FKParticle, FKScratch, M_ar1!, lG; 
        resampling = res, adaptive_threshold = adaptive_threshold)
        set_delta!(gstate.internal.scratch.par, Δ)
        B = [zeros(Int, N) for k = 1:T]
        nx = round(Int, 1/Δ)
        ind = 1:nx:T
        for k = 1:rep
            _run_smc!(gstate)
            L[j,k] = _log_likelihood(gstate)
            _estimate_smoothing!(out, f, gstate, B, ind)
            X[j,k,:] .= out
        end
    end
    L, X
end

function gather_experiment(data_prefix, data_postfix, AllResamplings, Δs, T_phys, rep)
    
    n_res = length(AllResamplings)
    n_Δs = length(Δs)
    d_f = Int(T_phys+1)
    
    L = zeros(n_res, n_Δs, rep)
    X = zeros(n_res, n_Δs, rep, d_f)
    for (i,res) in enumerate(keys(AllResamplings))
        out = load(string(data_prefix, "_$(res)_", data_postfix, ".jld2"))
        L[i,:,:], X[i,:,:,:] = out["L"], out["X"]        
    end
    L, X
end

function resampling_aliases(N)
    (
    Stratified = StratifiedResampling(N), 
    StratifiedPartition = StratifiedResampling(N; order=:partition),
    StratifiedAlternate = StratifiedResampling(N; order=:alternate),
    Systematic = SystematicResampling(N),
    SystematicPartition = SystematicResampling(N; order=:partition),
    SystematicAlternate = SystematicResampling(N; order=:alternate),
    SSP = SSPResampling(N),
    SSPPartition = SSPResampling(N; order=:partition),
    SSPAlternate = SSPResampling(N; order=:alternate),
    SSS = SSSResampling(N; order=:none),
    SSSPartition = SSSResampling(N; order=:partition),
    Killing = KillingResampling(N), 
    Multinomial = MultinomialResampling(N),
    Residual = ResidualResampling(N),
    )
end