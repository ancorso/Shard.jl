@with_kw mutable struct Sampler{P, V, T1 <: AbstractSpace, T2 <: AbstractSpace}
    mdp::P
    π::Policy
    s::V = rand(initialstate(mdp)) # Current State
    S::T1 = state_space(initial_observation(mdp, s)) # State space
    A::T2 = action_space(π) # Action space
    svec::AbstractArray = tovec(initial_observation(mdp, s), S) # Current observation
    max_steps::Int = 100
    required_columns::Array{Symbol} = π isa AdversarialPolicy ? [:x] : []
    γ::Float32 = discount(mdp)
    λ::Float32 = NaN32
    π_explore = nothing
    episode_length::Int64 = 0
    episode_checker::Function = (data, start, stop) -> true
end

Sampler(mdp, π; kwargs...) = Sampler(;mdp=mdp, π=π, kwargs...)

# Construct a vector of samplers from a vector of mdps
Sampler(mdps::AbstractVector, π; kwargs...) = [Sampler(mdps[i], π; kwargs...) for i in 1:length(mdps)]
        
function reset_sampler!(sampler::Sampler)
    sampler.s = rand(initialstate(sampler.mdp))
    sampler.svec = tovec(initial_observation(sampler.mdp, sampler.s), sampler.S)
    sampler.episode_length = 0
end

function initial_observation(mdp, s)
    if mdp isa POMDP
        return convert_o(AbstractArray, rand(initialobs(mdp, s)), mdp)
    else
        return convert_s(AbstractArray, s, mdp)
    end
end

function terminate_episode!(sampler::Sampler, data, j)
    haskey(data, :episode_end) && (data[:episode_end][1,j] = true)
    ep = j - sampler.episode_length + 1 : j
    haskey(data, :advantage) && fill_gae!(data, ep, sampler.π, sampler.λ, sampler.γ)
    haskey(data, :return) && fill_returns!(data, ep, sampler.γ)
    reset_sampler!(sampler)
end
    
function step!(data, j::Int, sampler::Sampler; explore=false, i=0)
    a, logprob = explore ? exploration(sampler.π_explore, sampler.svec, π_on=sampler.π, i=i) : (action(sampler.π, sampler.svec), NaN)
    length(a) == 1 && (a = a[1]) # actions always come out as an array
    
    # This implements the ability to get cost information from safety gym
    info = Dict()
    kwargs = haskey(data, :cost) ? (info=info,) : () 
    
    if sampler.π isa AdversarialPolicy
        π_adv = antagonist(sampler.π)
        π_adv_explore = sampler.π.A_explore
        x, xlogprob = explore ? exploration(π_adv_explore, sampler.svec, π_on=π_adv, i=i) : exploration(π_adv, sampler.svec)
        length(x) == 1 && (x = x[1]) # actions always come out as an array
        sp, r = gen(sampler.mdp, sampler.s, a, x; kwargs...)
        spvec = convert_s(AbstractArray, sp, sampler.mdp)
        data[:x][:, j:j] .= tovec(x, action_space(π_adv))
        haskey(data, :xlogprob) && (data[:xlogprob][:, j] .= xlogprob)
    elseif sampler.mdp isa POMDP
        sp, o, r = gen(sampler.mdp, sampler.s, a; kwargs...)
        spvec = convert_o(AbstractArray, o, sampler.mdp)
    else
        sp, r = gen(sampler.mdp, sampler.s, a; kwargs...)
        spvec = convert_s(AbstractArray, sp, sampler.mdp)
    end
    spvec = tovec(spvec, sampler.S)
    done = isterminal(sampler.mdp, sp)

    # Save the tuple
    bslice(data[:s], j:j) .= sampler.svec
    data[:a][:, j:j] .= tovec(a, sampler.A)
    bslice(data[:sp], j:j) .= spvec
    data[:r][1, j] = r
    data[:done][1, j] = done
    
    # Handle optional data storage
    haskey(data, :logprob) && (data[:logprob][:, j] .= logprob)  
    haskey(data, :t) && (data[:t][1, j] = sampler.episode_length + 1)
    haskey(data, :i) && (data[:i][1, j] = i+1)
    haskey(data, :cost) && (data[:cost][1,j] = info["cost"])
    haskey(data, :fail) && (data[:fail][1,j] = extra_functions["isfailure"](sampler.mdp, sp))
    
    # Cut the episode short if needed
    sampler.episode_length += 1
    if done || sampler.episode_length >= sampler.max_steps 
        terminate_episode!(sampler, data, j)
    else
        sampler.s = sp
        sampler.svec = spvec
    end
end

function steps!(sampler::Sampler; Nsteps = 1, explore=false, i=0, reset=false, return_episodes=false, return_at_episode_end=false)
    data = mdp_data(sampler.S, sampler.A, Nsteps, return_episodes ? [sampler.required_columns..., :episode_end] : sampler.required_columns)
    for j=1:Nsteps
        step!(data, j, sampler, explore=explore, i=i + (j-1))
        if return_at_episode_end && sampler.episode_length == 0 
            trim!(data, j)
            break
        end
    end
    reset && terminate_episode!(sampler, data, Nsteps)
    return_episodes ? (data, episodes(data)) : data
end

function steps!(samplers::Vector{T}; Nsteps = 1, explore = false, i = 0, reset = false, return_episodes = false) where {T<:Sampler}
    data = mdp_data(samplers[1].S, samplers[1].A, Nsteps*length(samplers), return_episodes ? [samplers[1].required_columns..., :episode_end] : samplers[1].required_columns)
    j = 1
    for s=1:Nsteps
        for sampler in samplers
            step!(data, j, sampler, explore = explore, i = i + (j-1))
            j += 1
        end
    end
    reset && terminate_episode!(sampler, data, Nsteps)
    return_episodes ? (data, episodes(data)) : data
end

function episodes!(sampler::Sampler; Neps=1, explore=false, i=0, return_episodes=false)
    reset_sampler!(sampler)
    data = mdp_data(sampler.S, sampler.A, Neps*sampler.max_steps, sampler.required_columns)
    episode_starts = zeros(Int, Neps)
    episode_ends = zeros(Int, Neps)
    j, k = 0, 1
    while k <= Neps
        episode_starts[k] = j+1
        while true
            j = j+1
            step!(data, j, sampler, explore=explore, i=i + (i-1))
            if sampler.episode_length == 0
                episode_ends[k] = j
                sampler.episode_checker(data, episode_starts[k], j) ? (k = k+1) : (j = episode_starts[k])
                break
            end
        end
    end
    trim!(data, j)
    return_episodes ? (data, zip(episode_starts, episode_ends)) : data
end

function fillto!(b::ExperienceBuffer, s::Union{Sampler, Vector{T}}, N::Int; i=1, explore=false) where {T <: Sampler}
    Nfill = max(0, N - length(b))
    Nfill > 0 && push!(b, steps!(s, i=i, Nsteps=Nfill, explore=explore))
    Nfill
end

## metric

# Recover multiple metrics from a single sampler
metrics_by_key(data, start, stop; keys) = [sum(data[key][1,start:stop]) for key in keys]

function metrics_by_key(s::Sampler; keys, Neps=100, kwargs...)
    data = episodes!(s, Neps=Neps; kwargs...)
    [sum(data[key]) / Neps for key in keys]
end

# recover a single metric 
metric_by_key(data, start, stop; key) = metrics_by_key(data, start, stop; keys=[key])[1]

metric_by_key(s::Sampler; key, Neps=100, kwargs...) = metrics_by_key(s; keys=[key], Neps=Neps, kwargs...)[1]

# Get the undiscounted return
undiscounted_return(data, start, stop) = metric_by_key(data, start, stop; key=:r)
undiscounted_return(s::Sampler; Neps=100, kwargs...) = metric_by_key(s; Neps=Neps, key=:r, kwargs...)


## Discounted returns
function discounted_return(data, start, stop, γ)
    r = 0f0
    for i in reverse(start:stop)
        r = data[:r][1, i] + γ*r
    end
    r
end

function discounted_return(s::Sampler; Neps=100, kwargs...)
    data, episodes = episodes!(s, Neps=Neps, return_episodes=true; kwargs...)
    mean([discounted_return(data, e..., discount(s.mdp)) for e in episodes])
end

## Failures
failure(data, start, stop; threshold = 0.) = undiscounted_return(data, start, stop) < threshold

function failure(s::Sampler; threshold=0., Neps=100, kwargs...)
    data, episodes = episodes!(s, Neps = Neps, return_episodes = true; kwargs...)
    mean([failure(data, e..., threshold = threshold) for e in episodes])
end
    
    
## Generalized Advantage Estimation
function fill_gae!(d::ExperienceBuffer, V, λ::Float32, γ::Float32)
    eps = episodes(d)
    for ep in eps
        fill_gae!(d, ep, V, λ, γ)
    end
end

function fill_gae!(d, episode_range, V, λ::Float32, γ::Float32)
    A, c = 0f0, λ*γ
    nd = ndims(d[:s])
    for i in reverse(episode_range)
        Vsp = value(V, bslice(d[:sp], i:i))
        Vs = value(V, bslice(d[:s], i:i))
        @assert length(Vs) == 1
        A = c*A + d[:r][1,i] + (1.f0 - d[:done][1,i])*γ*Vsp[1] - Vs[1]
        @assert !isnan(A)
        d[:advantage][:, i] .= A
    end
end

function fill_returns!(data, episode_range, γ::Float32)
    r = 0f0
    for i in reverse(episode_range)
        r = data[:r][1, i] + γ*r
        data[:return][:, i] .= r
    end
end

# Utils
function trim!(data::Dict{Symbol, Array}, N)
    for k in keys(data)
        data[k] = bslice(data[k], 1:N)
    end
    data
end

