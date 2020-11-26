@with_kw mutable struct Sampler
    mdp
    π::Policy
    max_steps::Int = 100
    exploration_policy::Union{ExplorationPolicy, Nothing} = nothing
    rng::AbstractRNG = Random.GLOBAL_RNG
    s = rand(rng, initialstate(mdp))
    svec::AbstractArray = convert_s(AbstractArray, s, mdp)
    episode_length::Int64 = 0
    episode_checker::Function = (data, start, stop) -> true
end

explore(s::Sampler) = !isnothing(s.exploration_policy)

function step!(data, j::Int, sampler::Sampler; i = 0, baseline = nothing)
    a = explore(sampler) ? action(sampler.exploration_policy, sampler.π, i, sampler.svec) : action(sampler.π, sampler.svec)
    sp, r = gen(sampler.mdp, sampler.s, a, sampler.rng)
    done = isterminal(sampler.mdp, sp)

    # Save the tuple
    spvec = convert_s(AbstractArray, sp, sampler.mdp)
    data[:s][:, j] .= sampler.svec
    data[:a][:, j] .= (a isa AbstractArray) ?  a : Flux.onehot(a, actions(sampler.mdp))
    data[:sp][:, j] .= spvec
    data[:r][1, j] = r
    data[:done][1, j] = done
    
    # Cut the episode short if needed
    sampler.episode_length += 1
    if done || sampler.episode_length >= sampler.max_steps 
        if !isnothing(baseline)
            start = j - sampler.episode_length
            fill_gae!(data, start, j, baseline.V, baseline.λ, γ)
            fill_returns!(data, start, j, γ)
        end
        sampler.s = rand(sampler.rng, initialstate(sampler.mdp))
        sampler.svec = convert_s(AbstractArray, sampler.s, sampler.mdp)
        sampler.episode_length = 0
    else
        sampler.s = sp
        sampler.svec = spvec
    end
end

function steps!(sampler::Sampler; Nsteps = 1, i = 0, baseline = nothing)
    data = mdp_data(sdim(sampler.mdp), adim(sampler.mdp), Nsteps, gae = !isnothing(baseline))
    for j=1:Nsteps
        step!(data, j, sampler, i = i + (j-1), baseline = baseline)
    end
    data
end

function episodes!(sampler::Sampler; Neps = 1, i = 0, baseline = nothing, return_episodes = false)
    sampler.episode_length = 0
    data = mdp_data(sdim(sampler.mdp), adim(sampler.mdp), Neps*sampler.max_steps, gae = !isnothing(baseline))
    episode_starts = zeros(Int, Neps)
    episode_ends = zeros(Int, Neps)
    j, k = 0, 1
    while k <= Neps
        episode_starts[k] = j+1
        while true
            j = j+1
            step!(data, j, sampler, i = i + (i-1), baseline = baseline)
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

function fillto!(b::ExperienceBuffer, s::Sampler, N::Int; i = 1)
    Nfill = max(0, N - length(b))
    Nfill > 0 && push!(b, steps!(s, i = i, Nsteps = Nfill))
    Nfill
end

## Undiscounted returns
undiscounted_return(data, start, stop) = sum(data[:r][1,start:stop])

function undiscounted_return(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG)
    s = Sampler(mdp = mdp, π = policy, max_steps = max_steps, rng = rng)
    data = episodes!(s, Neps = Neps)
    sum(data[:r]) / Neps
end

## Discounted returns
function discounted_return(data, start, stop, γ)
    r = 0f0
    for i in reverse(start:stop)
        r = data[:r][1, i] + γ*r
    end
    r
end

function discounted_return(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG)
    s = Sampler(mdp = mdp, π = policy, max_steps = max_steps, rng = rng)
    data, episodes = episodes!(s, Neps = Neps, return_episodes = true)
    mean([discounted_return(data, e..., discount(mdp)) for e in episodes])
end

## Failures
failure(data, start, stop; threshold = 0.) = undiscounted_return(data, start, stop) < threshold

function failure(mdp, policy; threshold = 0., Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG)
    s = Sampler(mdp = mdp, π = policy, max_steps = max_steps, rng = rng)
    data, episodes = episodes!(s, Neps = Neps, return_episodes = true)
    mean([failure(data, e..., threshold = threshold) for e in episodes])
end
    
    
## Generalized Advantage Estimation
function fill_gae!(data, start::Int, stop::Int, V, λ::Float32, γ::Float32)
    A, c = 0f0, λ*γ
    for i in reverse(start:stop)
        Vsp = V(b[:sp][:,i])
        Vs = V(b[:s][:,i])
        @assert length(Vs) == 1
        A = c*A + b[:r][1,i] + (1.f0 - b[:done][1,i])*γ*Vsp[1] - Vs[1]
        b[:advantage][:, i] .= A
    end
end

function fill_returns!(data, start::Int, stop::Int, γ::Float32)
    r = 0f0
    for i in reverse(start:stop)
        r = b[:r][1, i] + γ*r
        b[:return][:, i] .= r
    end
end

# Utils
function trim!(data::Dict{Symbol, Array{Float32, 2}}, N)
    for k in keys(data)
        data[k] = data[k][:, 1:N]
    end
    data
end
