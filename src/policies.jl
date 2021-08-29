abstract type NetworkPolicy <: Policy end

device(π::N) where N<:NetworkPolicy = π.device
actor(π::N) where N<:NetworkPolicy = π
critic(π::N) where N<:NetworkPolicy = π

# Call policies as functions calls the value function
(pol::NetworkPolicy)(x...) = value(pol, x...)

layers(π::Chain) = π.layers

function polyak_average!(to, from, τ=1f0)
    to_data = Flux.params(to).order.data
    from_data, from_device = Flux.params(from).order.data, device(from)
    device_match = from_device == device(to)
    for i = 1:length(to_data)
        if device_match
            copyto!(to_data[i], τ.*from_data[i] .+ (1f0-τ).*to_data[i])
        else
            copyto!(to_data[i], τ.*from_data[i] .+ (1f0-τ).*from_device(to_data[i]))
        end            
    end
end

function Base.copyto!(to, from)
    for i = 1:length(Flux.params(to).order.data)
        copyto!(Flux.params(to).order.data[i], Flux.params(from).order.data[i])
    end
end

## Network for representing continous functions (value or policy)
mutable struct ContinuousNetwork <: NetworkPolicy
    network
    output_dim
    device
    ContinuousNetwork(network, output_dim=size(last(network.layers).weight,1), dev=nothing) = new(network, output_dim, device(network))
end

Flux.@functor ContinuousNetwork 

Flux.trainable(π::ContinuousNetwork) = Flux.trainable(π.network)

layers(π::ContinuousNetwork) = π.network.layers

POMDPs.action(π::ContinuousNetwork, s) = value(π, s)

POMDPs.value(π::ContinuousNetwork, s) = mdcall(π.network, s, π.device)

POMDPs.value(π::ContinuousNetwork, s, a) = mdcall(π.network, vcat(s,a), π.device)

action_space(π::ContinuousNetwork) = ContinuousSpace(π.output_dim)



## Network for representing a discrete set of outputs (value or policy)
# NOTE: Incoming actions (i.e. arguments) are all assumed to be one hot encoding. Outputs are discrete actions taken form outputs
mutable struct DiscreteNetwork <: NetworkPolicy
    network
    outputs
    logit_conversion
    device
    DiscreteNetwork(network, outputs, logit_conversion=softmax, dev=nothing) = new(network, cpu(outputs), logit_conversion, device(network))
end

Flux.@functor DiscreteNetwork 

Flux.trainable(π::DiscreteNetwork) = Flux.trainable(π.network)

layers(π::DiscreteNetwork) = π.network.layers

POMDPs.value(π::DiscreteNetwork, s) = mdcall(π.network, s, π.device)

POMDPs.value(π::DiscreteNetwork, s, a_oh) = sum(value(π, s) .* a_oh, dims = 1)

POMDPs.action(π::DiscreteNetwork, s) = π.outputs[mapslices(argmax, value(π, s), dims=1)]

function Flux.onehotbatch(π::DiscreteNetwork, a)  
    ignore() do 
        a_oh = Flux.onehotbatch(a[:] |> cpu, π.outputs) |> device(a)
        length(a) == 1 ? dropdims(a_oh, dims=2) : a_oh
    end
end 

logits(π::DiscreteNetwork, s) = π.logit_conversion(value(π, s))

categorical_logpdf(probs, a_oh) = log.(sum(probs .* a_oh, dims = 1) .+ eps(Float32))

function exploration(π::DiscreteNetwork, s; kwargs...)
    ps = logits(π, s) 
    a = π.outputs[mapslices((v)->rand(Categorical(v)), ps, dims=1)]
    a, categorical_logpdf(ps, Flux.onehotbatch(π, a))
end

function Distributions.logpdf(π::DiscreteNetwork, s, a)
    Zygote.ignore() do 
        size(a,1) == 1 && (a = Flux.onehotbatch(π, a)) 
    end
    return categorical_logpdf(logits(π, s), a)
end

function Distributions.entropy(π::DiscreteNetwork, s)
    ps = logits(π, s)
    sum(ps .* log.(ps .+ eps(Float32)), dims=1)
end

action_space(π::DiscreteNetwork) = DiscreteSpace(length(π.outputs), π.outputs)



## Double Network architecture
mutable struct DoubleNetwork{T1, T2} <: NetworkPolicy
    N1::T1
    N2::T2
end

Flux.@functor DoubleNetwork

Flux.trainable(π::DoubleNetwork) = (Flux.trainable(π.N1)..., Flux.trainable(π.N2)...)

layers(π::DoubleNetwork) = unique((layers(π.N1)..., layers(π.N2)...))

device(π::DoubleNetwork) = device(π.N1) == device(π.N2) ? device(π.N1) : error("Mismatched devices")

POMDPs.value(π::DoubleNetwork, s) = (value(π.N1, s), value(π.N2, s))

POMDPs.value(π::DoubleNetwork, s, a) = (value(π.N1, s, a), value(π.N2, s, a))

POMDPs.action(π::DoubleNetwork, s) = (action(π.N1, s), action(π.N2, s))

exploration(π::DoubleNetwork, s; kwargs...) = (exploration(π.N1, s; kwargs...), exploration(π.N2, s; kwargs...))
    
Distributions.logpdf(π::DoubleNetwork, s, a) = (logpdf(π.N1, s, a), logpdf(π.N2, s, a))

Distributions.entropy(π::DoubleNetwork, s) = (entropy(π.N1, s), entropy(π.N2, s))

action_space(π::DoubleNetwork) = action_space(π.N1)


## Double Network architecture
mutable struct AdversarialPolicy{T1, T2} <: NetworkPolicy
    P::T1 # Protagonist
    A::T2 # Antagonist
    A_explore # exploration policy for antagonist
    AdversarialPolicy(P::T1, A::T2, A_explore=A) where {T1, T2} = new{T1, T2}(P, A, A_explore)
end
Flux.@functor AdversarialPolicy
Flux.trainable(π::AdversarialPolicy) = (Flux.trainable(π.P)..., Flux.trainable(π.A)...)
layers(π::AdversarialPolicy) = unique((layers(π.P)..., layers(π.A)...))
device(π::AdversarialPolicy) = device(π.A) == device(π.P) ? device(π.A) : error("Mismatched devices")
POMDPs.value(π::AdversarialPolicy, s) = value(π.P, s)
POMDPs.value(π::AdversarialPolicy, s, a) = value(π.P, s, a)
POMDPs.action(π::AdversarialPolicy, s) = action(π.P, s)
exploration(π::AdversarialPolicy, s; kwargs...) = exploration(π.P, s; kwargs...)
Distributions.logpdf(π::AdversarialPolicy, s, a) = logpdf(π.P, s, a)
Distributions.entropy(π::AdversarialPolicy, s) = entropy(π.P, s)
action_space(π::AdversarialPolicy) = action_space(π.P)

protagonist(π::AdversarialPolicy) = π.P
antagonist(π::AdversarialPolicy) = π.A


## Actor Critic Architecture
mutable struct ActorCritic{TA, TC} <: NetworkPolicy
    A::TA # actor 
    C::TC # critic
end

Flux.@functor ActorCritic

Flux.trainable(π::ActorCritic) = (Flux.trainable(π.A)..., Flux.trainable(π.C)...)

layers(π::ActorCritic) = unique((layers(π.A)..., layers(π.C)...))

device(π::ActorCritic) = device(π.A) == device(π.C) ? device(π.A) : error("Mismatched devices")

POMDPs.value(π::ActorCritic, s) = value(π.C, s)

POMDPs.value(π::ActorCritic, s, a) = value(π.C, s, a)

POMDPs.action(π::ActorCritic, s) = action(π.A, s)

exploration(π::ActorCritic, s; kwargs...) = exploration(π.A, s; kwargs...)
    
Distributions.logpdf(π::ActorCritic, s, a) = logpdf(π.A, s, a)

Distributions.entropy(π::ActorCritic, s) = entropy(π.A, s)

action_space(π::ActorCritic) = action_space(π.A)

actor(π::AC) where AC<:ActorCritic = π.A
critic(π::AC) where AC<:ActorCritic = π.C


## Gaussian Policy
mutable struct GaussianPolicy <: NetworkPolicy
    μ::ContinuousNetwork
    logΣ::AbstractArray
end

Flux.@functor GaussianPolicy

Flux.trainable(π::GaussianPolicy) = (Flux.trainable(π.μ)..., π.logΣ)

layers(π::GaussianPolicy) = (layers(π.μ)..., π.logΣ)

device(π::GaussianPolicy) = device(π.μ) == device(π.logΣ) ? device(π.μ) : error("Mismatched devices")

POMDPs.action(π::GaussianPolicy, s) = action(π.μ, s)

function gaussian_logpdf(μ, logΣ, a)
    σ² = exp.(logΣ).^2
    sum(-((a .- μ).^2) ./ (2 .* σ²) .-  0.9189385332046727f0 .- logΣ, dims = 1) # 0.9189385332046727f0 = log(sqrt(2π))
end 

function exploration(π::GaussianPolicy, s; kwargs...) 
    μ, logΣ = action(π, s), device(s)(π.logΣ)
    σ = exp.(logΣ)
    ϵ = Zygote.ignore(() -> randn(Float32, size(μ)...) |> device(s))
    a = ϵ.*σ .+ μ
    a, gaussian_logpdf(μ, logΣ, a)
end

Distributions.logpdf(π::GaussianPolicy, s, a) = gaussian_logpdf(action(π, s), device(s)(π.logΣ), a)

Distributions.entropy(π::GaussianPolicy, s) = 1.4189385332046727f0 .+ sum(device(s)(π.logΣ)) # 1.4189385332046727 = 0.5 + 0.5 * log(2π)

action_space(π::GaussianPolicy) = action_space(π.μ)


## Squashed Gaussian policy
mutable struct SquashedGaussianPolicy <: NetworkPolicy
    μ::ContinuousNetwork
    logΣ::ContinuousNetwork
    ascale::Float32
    SquashedGaussianPolicy(μ, logΣ, ascale=1f0) = new(μ, logΣ, ascale)
end

Flux.@functor SquashedGaussianPolicy

Flux.trainable(π::SquashedGaussianPolicy) = (Flux.trainable(π.μ)..., Flux.trainable(π.logΣ)...)

layers(π::SquashedGaussianPolicy) = unique((layers(π.μ)..., layers(π.logΣ)...))

device(π::SquashedGaussianPolicy) = device(π.μ) == device(π.logΣ) ? device(π.μ) : error("Mismatched devices")

POMDPs.action(π::SquashedGaussianPolicy, s) = π.ascale .* tanh.(π.μ(s))

function squashed_gaussian_σ(logΣ)
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2
    # logΣ = LOG_STD_MIN .+ 0.5f0 .* (LOG_STD_MAX - LOG_STD_MIN) * (logΣ .+ 1)
    logΣ = clamp.(logΣ, LOG_STD_MIN, LOG_STD_MAX)
    exp.(logΣ)
end

# a is  untanh'd 
function squashed_gaussian_logprob(μ, logΣ, a)
    σ² = squashed_gaussian_σ(logΣ).^2
    sum(-((a .- μ).^2) ./ (2 .* σ²) .- 0.9189385332046727f0 .- logΣ .- 2*(log(2f0) .- a .- softplus.(-2 .* a)), dims=1)
end

function exploration(π::SquashedGaussianPolicy, s; kwargs...)
    μ, logΣ = π.μ(s), π.logΣ(s)
    σ = squashed_gaussian_σ(logΣ)
    ϵ = Zygote.ignore(() -> randn(Float32, size(μ)...) |> device(s))
    a_pretanh = ϵ.*σ .+ μ
    π.ascale .* tanh.(a_pretanh), squashed_gaussian_logprob(μ ,logΣ, a_pretanh)
end

Distributions.logpdf(π::SquashedGaussianPolicy, s, a) = squashed_gaussian_logprob(π.μ(s), π.logΣ(s), atanh.(clamp.(a ./ π.ascale, -1f0 + 1f-5, 1f0 - 1f-5)))

Distributions.entropy(π::SquashedGaussianPolicy, s) = 1.4189385332046727f0 .+ sum(value(π.logΣ, s), dims=1) # 1.4189385332046727 = 0.5 + 0.5 * log(2π) #TODO: This doesn't account for the squash

action_space(π::SquashedGaussianPolicy) = action_space(π.μ)

## Distribution policy -> For state-independent policies
mutable struct DistributionPolicy <: Policy
    distribution
end

layers(π::DistributionPolicy) = ()

device(π::DistributionPolicy) = cpu
actor(π::DistributionPolicy) = π

function exploration(π::DistributionPolicy, s; kwargs...)
    B = ndims(s) > 1 ? (size(s)[end]) : () # NOTE: this is a hack and doesn't work if the state space has more than one dim
    a = Float32.(rand(π.distribution, B...))
    a |> device(s), logpdf(π, s, a)
end

Distributions.logpdf(π::DistributionPolicy, s, a) = Float32.(reshape([logpdf(π.distribution, a)...], 1, :)) |> device(s)

function Distributions.entropy(π::DistributionPolicy, s)
    B = ndims(s) > 1 ? (size(s)[end]) : () # NOTE: this is a hack and doesn't work if the state space has more than one dim
    Float32(entropy(π.distribution)) * ones(Float32, 1, B...)
end

action_space(π::DistributionPolicy) = ContinuousSpace(length(π.distribution))


## Exploration policy with Gaussian noise
struct LinearDecaySchedule{R<:Real} <: Function
    start::R
    stop::R
    steps::Int
end

function (schedule::LinearDecaySchedule)(i)
    rate = (schedule.start - schedule.stop) / schedule.steps
    val = schedule.start - i*rate 
    val = max(schedule.stop, val)
end

mutable struct ϵGreedyPolicy <: Policy
    ϵ::Function
    actions
end

ϵGreedyPolicy(ϵ::Real, actions) = ϵGreedyPolicy((i) -> ϵ, actions)

function exploration(π::ϵGreedyPolicy, s; π_on, i,)
    rand() < π.ϵ(i) ? (rand(π.actions, 1), NaN) : (action(π_on, s), NaN)
end

## Mixing policy
mutable struct MixedPolicy <: Policy
    ϵ::Function
    policy
end

MixedPolicy(ϵ::Real, policy) = MixedPolicy((i) -> ϵ, policy)

function exploration(π::MixedPolicy, s; π_on, i,)
    ϵ = π.ϵ(i)
    x = rand() < ϵ ? rand(π.policy) : exploration(π_on, s; i=i, π_on=π_on)[1]
    size(x) == () && (x = [x]) # actions always come out as an array
    logp1 = Base.log(ϵ) .+ logpdf.(π.policy, x)
    logp2 = Base.log(1-ϵ) .+ logpdf(π_on, s, x)
    
    x, logsumexp(vcat(logp1, logp2), dims=1)
end



## Exploration policy with Gaussian noise
@with_kw mutable struct GaussianNoiseExplorationPolicy <: Policy
    σ::Function = (i) -> 0.01f0
    a_min::Vector{Float32} = [-Inf32]
    a_max::Vector{Float32} = [Inf32]
    ϵ_min::Float32 = -Inf32
    ϵ_max::Float32 = Inf32
end

GaussianNoiseExplorationPolicy(σ::Real; kwargs...) = GaussianNoiseExplorationPolicy(σ = (i) -> σ; kwargs...)
GaussianNoiseExplorationPolicy(σ::Function; kwargs...) = GaussianNoiseExplorationPolicy(σ = σ; kwargs...)

function exploration(π::GaussianNoiseExplorationPolicy, s; π_on, i)
    a = action(π_on, s) |> cpu
    ϵ = randn(Float32, size(a)...)*π.σ(i)
    clamp.(a .+ clamp.(ϵ, π.ϵ_min, π.ϵ_max), π.a_min, π.a_max) |> device(s), NaN
end


## use exploration policy for first N timesteps, then revert to base policy
@with_kw mutable struct FirstExplorePolicy <: Policy
    N::Int64 # Number of steps to explore for
    initial_policy::Policy # the policy to use for the first N steps
    after_policy::Union{Nothing, Policy} = nothing # the policy to use after the first N steps. Nothing means you will use on-policy
end

FirstExplorePolicy(N::Int64, initial_policy::Policy) = FirstExplorePolicy(N, initial_policy, nothing)

function exploration(π::FirstExplorePolicy, s; π_on, i)
    if i < π.N
        return action(π.initial_policy, s), NaN
    elseif isnothing(π.after_policy)
        return action(π_on, s), NaN
    else
        return exploration(π.after_policy, s, π_on=π_on, i=i)
    end
end

