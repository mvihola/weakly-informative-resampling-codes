using Random, LinearAlgebra
using Statistics: mean

# Auxiliary data for simple Resamplings
struct Resampling{T,O,FT <: AbstractFloat, R, IT, OT, AT, PT}
    u::Vector{FT}
    ind::IT
    aux::AT
    par::PT
    order::OT
end

# If randomize is :circular or :shuffle, the output of the resampling
# is randomized so that a random circular shift (:circular) or random
# permutation (:shuffle) is applied to the indices
function SimpleResampling(T, n::Integer; FT::DataType=Float64, aux=nothing, 
    order=:none, randomize=:none, args...)
    IntT = typeof(n)
    par = (; args...)
    PT = typeof(par)
    AT = typeof(aux)
    if order == :none
        ord = Base.OneTo(n)
    elseif order in (:sort_state, :sort, :partition, :alternate)
        ord = IntT.(collect(1:n))
    else
        error("Invalid argument order=$(order)")
    end
    if T in (:multinomial, :systematic, :stratified) && randomize != :circular && order != :alternate
        ind = nothing
    else
        ind = zeros(IntT,n)
    end
    OT = typeof(ord)
    IT = typeof(ind)
    Resampling{T,order,FT,randomize,IT,OT,AT,PT}(zeros(FT,n), ind, aux, par, ord)
end
# Simple multinomial, Killing, Systematic, Stratified:
MultinomialResampling(n::Integer; args...) = SimpleResampling(:multinomial, n; args...)
KillingResampling(n::Integer; max_div=true, args...) = SimpleResampling(:killing, n; max_div=max_div, args...)
SystematicResampling(n::Integer; args...) = SimpleResampling(:systematic, n; args...)
StratifiedResampling(n::Integer; args...) = SimpleResampling(:stratified, n; args...)

function AuxWeightResampling(T, n::Integer; FT::DataType=Float64, args...)
    aux = (w = zeros(FT, n),)
    SimpleResampling(T, n; FT=FT, aux=aux, args...)
end
ResidualResampling(n::Integer; args...) = AuxWeightResampling(:residual, n; args...)
SSPResampling(n::Integer; args...) = AuxWeightResampling(:ssp, n; args...)
# Simple limit systematic if condition is met; otherwise systematic:
SSSResampling(n::Integer; order=:partition, args...) = AuxWeightResampling(:sss, n; order=order, args...)

@inline function resample!(res, ind, w; rng=Random.GLOBAL_RNG, state=nothing)
    _set_resample_order!(res, state, w)
    _resample!(res, ind, w, rng)
    _randomize!(res, ind, rng)
    nothing
end

# Resampling for all (called helpers below)
function _resample!(res::Resampling{:multinomial}, ind, w, rng)
    _generate_u_multinomial!(res.u, rng)
    _find_matching_indices!(ind, w, res.u, res.order)
    nothing
end
function _resample!(res::Resampling{:stratified}, ind, w, rng)
    _generate_u_stratified!(res.u, rng)
    _find_matching_indices!(ind, w, res.u, res.order)
    nothing
end
function _resample!(res::Resampling{:systematic}, ind, w, rng)
    _generate_u_systematic!(res.u, rng)
    _find_matching_indices!(ind, w, res.u, res.order)
    nothing
end
function _resample!(res::Resampling{:sss}, ind, w, rng)
    n = length(res.u)
    c = _sss_condition(w)
    if c*n <= 1
        # Condition is met, do single-event stuff
        _sss_resample(ind, w, rng, res.u, res.aux.w, c)
    else
        # Stanrdard systematic
        _generate_u_systematic!(res.u, rng)
        _find_matching_indices!(ind, w, res.u, res.order)
    end
    nothing
end
@inline function _sss_condition(w, wm=1/length(w))
    c = mapreduce(w_i -> abs(w_i - wm), +, w)/2
    return c
end
function _sss_resample(ind, w, rng, w_small, w_large, c::FT) where {FT <: AbstractFloat}
    ind .= eachindex(ind)
    # Whether there is a resample event:
    if rand(rng) >= c*length(w)
        return nothing
    end
    # Form weights:
    inv_c = one(FT)/c
    wm = 1/length(w)
    for i in eachindex(w)
        @inbounds if w[i] > wm
            w_large[i] = (w[i] - wm) * inv_c
            w_small[i] = zero(FT)
        else
            w_small[i] = (wm - w[i]) * inv_c
            w_large[i] = zero(FT)
        end
    end
    k = _find_matching_index(w_small, rand(rng))
    i = _find_matching_index(w_large, rand(rng))
    @inbounds ind[k] = i
end

function _resample!(res::Resampling{:residual}, ind, w, rng)
    n = length(ind)
    copyto!(res.aux.w, w)
    res.aux.w .*= n
    k = 0
    for i = 1:n
        @inbounds r_i = floor(res.aux.w[i])
        @inbounds res.aux.w[i] -= r_i
        r_int = Int(r_i)
        if r_int > 0
            @assert k + r_int <= n
            @inbounds ind[(k+1):(k+r_int)] .= i
            k += r_int
        end
    end
    res.aux.w ./= sum(res.aux.w)
    r = n - k
    if r > 0
        _generate_u_multinomial!(view(res.u, 1:r), rng)
        _find_matching_indices!(view(ind, (k+1):n), res.aux.w, res.u)
    end
    nothing
end
function _resample!(res::Resampling{:killing,O,FT}, ind, w, rng) where {O,FT}
    n = length(res.u)
    @assert length(ind) == n && length(w) == n
    w_max = res.par.max_div ? maximum(w) : one(FT)
    r = 0
    rand!(rng, res.u)
    for i = eachindex(w)
        @inbounds if res.u[i] <= w[i]/w_max
            ind[i] = i
        else
            ind[i] = 0
            r += 1
        end
    end
    if r > 0
        _generate_u_multinomial!(view(res.u, 1:r), rng)
        vi = view(res.ind, 1:r)
        _find_matching_indices!(vi, w, res.u)
        shuffle!(rng, vi)
        k = 0
        for i = eachindex(w)
            @inbounds if ind[i] == 0
                ind[i] = vi[k += 1]
            end
        end
    end
    w_max
end
function _resample!(res::Resampling{:ssp,O,FT}, ind, w, rng) where {O,FT}
    # Cheap Julia convert from https://github.com/nchopin/particles/blob/master/particles/resampling.py
    n = length(res.ind)
    @assert n == length(w)
    @assert n == length(ind)
    # Aliases for temporary vectors
    nr_children = res.ind; xi = res.aux.w; order = res.order
    xi .= n .* w
    nr_children .= floor.(xi)
    xi .-= nr_children
    rand!(rng, res.u)
    #i, j = 1, 2
    i, j = order[1], order[2]
    for k in 1:(n - 1)
        @inbounds delta_i = min(xi[j], one(FT) - xi[i])  # increase i, decr j
        @inbounds delta_j = min(xi[i], one(FT) - xi[j])  # the opposite
        sum_delta = delta_i + delta_j
        # prob we increase xi[i], decrease xi[j]
        pj = (sum_delta > zero(FT)) ? delta_i / sum_delta : zero(FT)
        # sum_delta = 0. => xi[i] = xi[j] = 0.
        if @inbounds res.u[k] < pj  # swap i, j, so that we always inc i
            j, i = i, j
            delta_i = delta_j
        end
        @inbounds if xi[j] < one(FT) - xi[i]
            @inbounds xi[i] += delta_i
            #j = k + 2
            j = (k == n-1) ? n+1 : order[k+2]
        else
            @inbounds xi[j] -= delta_i
            @inbounds nr_children[i] += 1
            #i = k + 2
            i = (k == n-1) ? n+1 : order[k+2]
        end
    end
    allocated = sum(nr_children)
    # due to round-off error accumulation, we may be missing one particle
    if allocated == n - 1
        last_ij = (j == n + 1) ? i : j
        @inbounds nr_children[last_ij] += 1
        allocated += 1
    end
    @assert allocated == n
    @inbounds repeat_indices!(ind, nr_children)
    nothing
end
Base.@propagate_inbounds function repeat_indices!(ind, nr_children)
    k = 1
    for i = 1:length(nr_children)
        nr_children[i] == 0 && continue
        end_k = k + nr_children[i] - 1
        for j = k:end_k
            ind[j] = i
        end
        k = end_k + 1
    end
    nothing    
end

function _sss_positive_negative_weights!(w_large, w_small, w, i, r_pp::FT, small_norm, large_norm) where {FT}
    # Mean weight
    wm = one(FT)/length(w)
    for j in eachindex(w)
        @inbounds if w[j] > wm
            w_large[j] = (j == i) ? r_pp : (w[j] - wm)*large_norm
            w_small[j] = zero(FT)
        else
            w_small[j] = (j == i) ? zero(FT) : (wm - w[j])*small_norm
            w_large[j] = zero(FT)
        end
    end
    nothing
end

_set_resample_order!(::Resampling{T,:none}, x, w) where {T} = nothing
function _set_resample_order!(res::Resampling{T,:sort_state}, x, w) where {T}
    sortperm!(res.order, x)
    nothing
end
function _set_resample_order!(res::Resampling{T,:sort}, x, w) where {T}
    sortperm!(res.order, w)
    nothing
end
function _set_resample_order!(res::Resampling{T,:partition}, x, w) where {T}
    _partition_order!(res.order, w)
    nothing
end
function _set_resample_order!(res::Resampling{T,:alternate}, x, w) where {T}
    pivot = 1/length(w)
    m = _partition_order!(res.ind, w, pivot)
    _alternate_order!(res.order, res.ind, w, pivot, m)
    nothing
end



_randomize!(::Resampling{T,O,FT,:none}, ind, rng) where {T,O,FT} = nothing
@inline function _randomize!(res::Resampling{T,O,FT,:circular}, ind, rng) where {T,O,FT <: AbstractFloat}
    n = length(res.order)
    r = floor(rand(rng, FT)*n)
    copyto!(res.ind, ind)
    circshift!(ind, res.ind, r)
    nothing
end
@inline function _randomize!(res::Resampling{T,O,FT,:shuffle}, ind, rng) where {T,O,FT <: AbstractFloat}
    shuffle!(rng, ind)
    nothing
end

function _generate_u_multinomial!(u::AbstractVector{FT}, rng) where {FT <: AbstractFloat}
    randexp!(rng, u)
    cumsum!(u, u)
    u ./= (u[end] + randexp(FT))
    nothing
end
function _generate_u_systematic!(u::AbstractVector{FT}, rng) where {FT <: AbstractFloat}
    u_ = rand(rng, FT)
    _fill_u_systematic!(u, u_)
    nothing
end
function _fill_u_systematic!(u, u_)
    n = length(u)
    for i = eachindex(u)
        @inbounds u[i] = (i-1+u_)/n
    end
    nothing
end
function _generate_u_stratified!(u, rng)
    n = length(u)
    rand!(rng, u)
    for i = eachindex(u)
        @inbounds u[i] = (i-1+u[i])/n
    end
    nothing
end
# Inverse CDF lookup: ind = F^{-1}(u), where F is cdf corresponding to p,
# built up 
# NB: u assumed to be in ascending order.
function _find_matching_indices!(ind, p, u, order=Base.OneTo(length(p)))
    n = length(ind)
    m = length(p)
    @assert m >= 1
    K = 1; @inbounds S = p[order[1]]
    for j in Base.OneTo(n)
        @inbounds U = u[j]
        # Find K such that F(K) >= u
        while K < m && U > S
            K = K + 1       # Note that K is not reset!
            @inbounds S = S + p[order[K]] # S is the partial sum up to K
        end
        #@inbounds ind[j] = order[K]
        @inbounds ind[order[j]] = order[K]
    end
    nothing
end

# Inverse cdf discrete random variable generation I~p with U~U(0,1)
function _find_matching_index(p, U)
    m = length(p)
    @assert m >= 1
    K = 1; @inbounds S = p[1]
    while K < m && U > S
        K = K + 1       # Note that K is not reset!
        @inbounds S = S + p[K] # S is the partial sum up to K
    end
    return K
end


Base.@propagate_inbounds function swapindices!(x, i, j)
    x[i], x[j] = x[j], x[i]
    nothing
end

# This finds order of partition so that first are those
# less than the pivot (default: average) and then those greather than the pivot
# (Like the partition scheme you do in quicksort)
function _partition_order!(ind, w, pivot=1/length(w))
    # Sanity checks
    n = length(w)
    n <= 1 && return nothing
    @assert n == length(ind)
    
    # Set order
    ind .= 1:n
    
    # Indices "where we are at" in rearrangement.
    # Start from the very left and very right end
    i_lower = 0; i_upper = n + 1
    
    while true
        # Find "next" index i_lower with > pivot:
        while i_lower < min(i_upper, n)
            i_lower += 1
            @inbounds w[ind[i_lower]] > pivot && break
        end
        # Find "previous" index i_upper with < pivot:
        while i_upper > i_lower
            i_upper -= 1
            @inbounds w[ind[i_upper]] < pivot && break
        end
        # If the indices met, we are done
        i_upper == i_lower && break
        # Otherwise, swap the elements
        @inbounds swapindices!(ind, i_lower, i_upper)
    end
    i_upper
end

# Assuming that ind is partition order for w, that is:
# [ind[1:m-1]] <= pivot & w[ind[m:end] >= pivot],
# this finds order so that max_k |sum(w[order[1:k] .- pivot)|
# is 'heuristically minimised' (not minimum, but something like that)
function _alternate_order!(order, ind, w, pivot::FT, m) where {FT <: AbstractFloat}
    n = length(w)
    @assert 1 < m <= n
    @assert length(order) == length(ind) == n
    
    i = 1; j = m
    s = zero(FT)
    for k = 1:n
        if (i < m && s >= zero(FT)) || j > n
            @inbounds order[k] = ind[i]
            @inbounds s += w[ind[i]] - pivot
            i += 1
        else # (i >= m || s < zero(FT)) && j <= n
            @inbounds order[k] = ind[j]
            @inbounds s += w[ind[j]] - pivot
            j += 1
        end
    end
    nothing
end

############################################################
## Helpers not required by the core functionality:
############################################################

# Set normalised weights that correspond to given log weights
function set_log_weights!(w, log_w)
    @assert length(log_w) == length(w)
    m_w = maximum(log_w)
    w .= exp.(log_w .- m_w)
    w ./= sum(w)
    nothing
end
