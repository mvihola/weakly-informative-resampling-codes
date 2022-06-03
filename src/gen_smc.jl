using Random, Statistics

include("resampling.jl")

# A "generic" sequential Monte Carlo implementation, with interfaces for
# AdaptiveParticleMCMC.jl

struct GenSMCState{
    ParticleT,
    ParticleS,
    spT <: Function,
    MT <: Function,
    lGT <: Function,
    lMT <: Union{Function,Nothing},
    internalT, 
    IT <: Integer}
    ###
    Particle::ParticleT
    ParticleScratch::ParticleS
    T::IT
    N::IT
    M!::MT
    lG::lGT
    lM::lMT
    set_param!::spT
    internal::internalT
end

function GenSMCState(T::Int, N::Int, ParticleType, ScratchType,
    M!::Function, lG::Function;
    set_param!::Function=(scratch, new_par) -> nothing,
    lM::Union{Function,Nothing} = nothing,
    resampling = MultinomialResampling(N),
    adaptive_threshold = nothing)
    
    @assert T>=1 "T must be at least 1"
    @assert N>=2 "N must be at least 2"
    
    internal = (
    # All particless
    X = [[ParticleType() for i=1:N] for j=1:T],
    # All ancestor indices
    A = [ones(Int, N) for i=1:T-1],
    # All unnormalised log weights
    lW = [zeros(N) for i=1:T],
    # All normalised weights
    W = [ones(N) for i=1:T],
    # Temporary weight vector (for bs)
    w = ones(N),
    # Where uniforms are sampled
    u = zeros(N),
    # Reference indices
    ref = ones(Int, T),
    # Reference state
    ref_state = [ParticleType() for i=1:T],
    # (Partial) log-normalising constants
    logZhats = zeros(T),
    # Instantiation of Scratch
    scratch = ScratchType(),
    # Random number generator
    rng = Random.GLOBAL_RNG,
    # Resampling strategy
    resampling = resampling,
    # Adaptive resampling threshold (if any)
    adaptive_threshold = adaptive_threshold
    )
    GenSMCState(ParticleType,ScratchType,T, N, M!, lG, lM,
    set_param!, internal)
end

function _normalise_log_weights!(w, logw) # Normalised probabilities from log weights ('log-sum-trick')
    m = maximum(logw)
    map!(lw -> exp(lw-m), w, logw)
    z = m + log(mean(w))
    sw = sum(w)
    if sw > zero(eltype(w))
        w ./= sw
    else
        w .= zero(eltype(w))
        @inbounds w[1] = one(eltype(w))
    end
    return z
end

"""
i = sample_from_categorical(p)

Direct implementation for drawing a random index i with probability p[i].
"""
function _sample_from_categorical(rng, p)
    U = rand(rng)
    K = 0; S = 0.0; m = length(p)
    while K < m && U >= S
        K = K + 1       # Note that K is not reset!
        @inbounds S = S + p[K] # S is the partial sum up to K
    end
    return K
end

# Stolen from SequentialMonteCarlo.jl particleCopy!
@inline @generated function copy_particle!(dest, src)
    fieldNames = fieldnames(dest)
    fieldTypes = dest.types
    numFields = length(fieldNames)
    expressions = Array{Expr}(undef, numFields)
    
    for i = 1:numFields
        fieldName = fieldNames[i]
        fieldType = fieldTypes[i]
        @assert isbitstype(fieldType) || hasmethod(copyto!, (fieldType,
        fieldType)) "$fieldName::$fieldType : copyto! must exist for isbitstype Particle fields"
        if !isbitstype(fieldType)
            @inbounds expressions[i] = :(copyto!(dest.$fieldName, src.$fieldName))
        else
            @inbounds expressions[i] = :(dest.$fieldName = src.$fieldName)
        end
    end
    body = Expr(:block, expressions...)
    
    quote
        # $(Expr(:meta, :inline))
        $body
        return
    end
end

@inline function effective_sample_size(w)
    one(eltype(w))/(length(w)*mapreduce(x -> x^2, +, w))
end
function _run_smc!(state::GenSMCState, conditional=false)
    adaptive_resampling = !isnothing(state.internal.adaptive_threshold)
    if adaptive_resampling        
        @assert !conditional "Conditional SMC not implemented with adaptive resampling"
        threshold = state.internal.adaptive_threshold
        @assert isreal(threshold) "Adaptive resampling threshold must be real"
        id = Base.OneTo(state.N)
        log_N = log(state.N)
    end
    scratch = state.internal.scratch
    X = state.internal.X; A = state.internal.A
    W = state.internal.W; lW = state.internal.lW
    ref = state.internal.ref
    logZhats = state.internal.logZhats
    rng = state.internal.rng; res = state.internal.resampling
    @inbounds x = X[1]
    for i = eachindex(x)
        @inbounds if !conditional || ref[1] != i
            state.M!(x[i], rng, 1, nothing, scratch)
        end
        @inbounds lW[1][i] = state.lG(1, x[i], scratch)
    end
    @inbounds logZhats[1] = _normalise_log_weights!(W[1], lW[1])
    for k = 2:state.T
        @inbounds a_ = A[k-1]; x = X[k]; x_ = X[k-1]; w_ = W[k-1]
        if conditional
            @inbounds conditional_resample!(res, a_, w_, ref[k], ref[k-1]; rng=rng, state=x_)
        else
            if adaptive_resampling && effective_sample_size(w_) >= threshold
                resampled = false
                a_ .= id
            else
                @inbounds resample!(res, a_, w_; rng=rng, state=x_)
                resampled = true
            end
        end
        for i = eachindex(x)
            if !conditional || ref[k] != i
                @inbounds state.M!(x[i], rng, k, x_[a_[i]], scratch)
            end
            @inbounds lW_ = state.lG(k, x[i], scratch)
            if !resampled
                @inbounds lW_ += log(w_[i]) + log_N
            end
            @inbounds lW[k][i] = lW_
        end
        @inbounds logZhats[k] = logZhats[k-1] + _normalise_log_weights!(W[k], lW[k])
    end
    nothing
end

@inline function _log_likelihood(state::GenSMCState)
    state.internal.logZhats[end]
end
@inline function _pick_particle!(state::GenSMCState)
    T = state.T; A = state.internal.A; ref = state.internal.ref
    @inbounds b = _sample_from_categorical(state.internal.rng,
    state.internal.W[T])
    @inbounds ref[T] = b
    @inbounds for k = (T-1):-1:1
        b = A[k][b]
        ref[k] = b
    end
    nothing
end
@inline function _save_reference!(state::GenSMCState)
    X = state.internal.X; ref = state.internal.ref
    ref_state = state.internal.ref_state
    for k = eachindex(ref_state)
        @inbounds copy_particle!(ref_state[k], X[k][ref[k]])
    end
end
@inline function _copy_reference!(out, state::GenSMCState)
    @inbounds for i = 1:length(out)
        copy_particle!(out[i], state.internal.ref_state[i])
    end
    nothing
end
@inline function _run_csmc!(state::GenSMCState)
    _run_smc!(state, true)
end
function _pick_particle_bs!(state::GenSMCState)
    w = state.internal.w; rng = state.internal.rng
    scratch = state.internal.scratch
    ref = state.internal.ref; ref_state = state.internal.ref_state
    T = state.T; X = state.internal.X
    b = _sample_from_categorical(rng, state.internal.W[T])
    ref[T] = b
    for k = state.T-1:-1:1
        @inbounds lW = state.internal.lW[k]
        @inbounds x = X[k+1][b]
        @inbounds x_ = X[k]
        @inbounds for i = 1:state.N
            w[i] = lW[i] + state.lM(k, x_[i], x, scratch)
        end
        normalise_log_weights!(w, w)
        b = _sample_from_categorical(rng, w)
        @inbounds state.internal.ref[k] = b
    end
    nothing
end

function _ancestor_tracing!(B, state::GenSMCState)
    T = state.T; A = state.internal.A
    for i = 1:state.N
        @inbounds B[T][i] = i
    end
    for k = state.T-1:-1:1
        for i = 1:state.N
            @inbounds B[k][i] = A[k][B[k+1][i]]
        end
    end
    nothing
end
function alive(state::GenSMCState)
    B = [zeros(Int, state.N) for k = 1:state.T]
    _ancestor_tracing!(B, state)
    length.(unique.(B))
end

function estimate_smoothing(state::GenSMCState, f::Function, ind=Base.OneTo(state.T))
    out = [f(state.internal.X[k][1]) for k = 1:state.T]
    B = [zeros(Int, state.N) for k = 1:state.T]
    _estimate_smoothing!(out, f, state, B, ind)
    out
end
function _estimate_smoothing!(out, f, state, B, ind=Base.OneTo(state.T))
    _ancestor_tracing!(B, state)
    w = state.internal.W[end]
    out .= 0.0
    for (j, k) in enumerate(ind)
        for i = Base.OneTo(state.N)
            @inbounds b_i = B[k][i]
            @inbounds x_i = state.internal.X[k][b_i]
            @inbounds out[j] += w[i] * f(x_i)
        end
    end
    nothing
end

_set_model_param!(gstate::GenSMCState, θ) = gstate.set_param!(gstate.internal.scratch, θ)
@inline function _init_path_storage(state::GenSMCState, nsim)
    [[state.Particle() for i=1:state.T] for j=1:nsim]
end