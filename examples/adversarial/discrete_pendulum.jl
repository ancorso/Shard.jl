using POMDPs, Crux, Flux, POMDPGym, Random, Distributions, POMDPPolicies, Plots
Crux.set_function("isfailure", POMDPGym.isfailure)

# Define the disturbance distribution based on a normal distribution
xnom = Normal(0f0, 0.5f0)
xs = Float32[-2., -0.5, 0, 0.5, 2.]
ps = exp.([logpdf(xnom, x) for x in xs])
ps ./= sum(ps)
px = DiscreteNonParametric(xs, ps)
xlogprobs = Base.log.(ps)

# Action space of the protagonist
as = Float32[-2., -0.5, 0, 0.5, 2.]

# Construct the MDP
mdp = AdditiveAdversarialMDP(InvertedPendulumMDP(actions=as, λcost=0, include_time_in_state=true), px)
S = state_space(mdp)

# construct the model 
QS(outputs) = DiscreteNetwork(Chain(Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, length(outputs))), outputs)
Pf(outputs) = DiscreteNetwork(Chain(Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, length(outputs), (x)->-softplus(-(x-2)))), outputs)
# Pf(outputs) = DiscreteNetwork(Chain(Dense(3, 256, tanh), Dense(256, 256, tanh), Dense(256, length(outputs), sigmoid)), outputs, (x) -> x ./ sum(x, dims=1))
AdvPol(protag = QS(as)) = AdversarialPolicy(protag, Pf(xs))

# Solve with DQN
𝒮_dqn = DQN(π=QS(as), S=S, N=50000, buffer_size=Int(1e4), buffer_init=1000, required_columns=[:fail])
π_dqn = solve(𝒮_dqn, mdp)

# show the nominal distriubtion of paths
D = steps!(Sampler(mdp, 𝒮_isarl.π.P, S=S, required_columns=[:fail]), Nsteps=10000)
D2 = steps!(Sampler(mdp, π_dqn, S=S, required_columns=[:fail]), Nsteps=10000)
scatter(D[:s][1, :], D[:s][3, :], marker_z = D[:done][1,:], xlabel="θ", ylabel="t", alpha=0.5)
scatter(D2[:s][1, :], D2[:s][3, :], marker_z = D2[:done][1,:], xlabel="θ", ylabel="t", alpha=0.5)

# solve with IS
𝒮_isarl = ISARL_Discrete(π=AdvPol(), 
                         S=S, 
                         ϵ_init = 1e-5,
                         N=100000, 
                         xlogprobs=xlogprobs, 
                         px=px, 
                         buffer_size=100_000,
                         buffer_init=1000, 
                         c_opt = (;batch_size=128),
                         x_c_opt=(;batch_size=1024), 
                         π_explore=ϵGreedyPolicy(Crux.LinearDecaySchedule(1f0, 0.1f0, 20000), as),
                         )
π_isarl = solve(𝒮_isarl, mdp)

y = Crux.IS_DQN_target(antagonist(𝒮_isarl.π),𝒮_isarl.𝒫, 𝒮_isarl.buffer, 1f0)

indices = 𝒮_isarl.buffer[:done][:]
y[indices]

value(antagonist(𝒮_isarl.π), 𝒮_isarl.buffer[:s][:,indices], 𝒮_isarl.buffer[:x][:,indices])


# check the number of failures in the buffers (compared to successs)
sum(𝒮_isarl.buffer[:fail])
sum(𝒮_isarl.buffer[:done])

# Plot the distribution of disturbances
histogram(Flux.onecold(𝒮_isarl.buffer[:x]))

# Plot the value function in 2D
v(θ, t) = sum(ps .* value(antagonist(𝒮_isarl.π), [θ, -1, t]))
heatmap(deg2rad(-20):0.01:deg2rad(20), 0:0.1:5, v)

# Show the distribution of data in the replay buffer
scatter(𝒮_isarl.buffer[:s][1, :], 𝒮_isarl.buffer[:s][3, :], marker_z = 𝒮_isarl.buffer[:done][1,:], xlabel="θ", ylabel="t", alpha=0.5)
vline!([deg2rad(20), deg2rad(-20)])


# pfail_dqn = Crux.failure(Sampler(mdp, π_dqn, S=S, max_steps=100), Neps=Int(1e5), threshold=100)
# println("DQN Failure rate: ", pfail_dqn)

# solve with RARL
# 𝒮_rarl = RARL(π=AdvPol(), S=S, N=N)
# π_rarl = solve(𝒮_rarl, mdp)

# pfail_rarl = Crux.failure(Sampler(mdp, protagonist(π_rarl), S=S, max_steps=100), Neps=Int(1e5), threshold=100)
# println("RARL Failure rate: ", pfail_rarl)



# pfail_isarl = Crux.failure(Sampler(mdp, protagonist(π_isarl), S=S, max_steps=100), Neps=Int(1e5), threshold=100)
# println("IS Failure rate: ", pfail_isarl)


