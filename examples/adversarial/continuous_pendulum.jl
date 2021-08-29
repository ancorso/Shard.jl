using POMDPs, Crux, Flux, POMDPGym, Random, Distributions, POMDPPolicies
Crux.set_function("isfailure", POMDPGym.isfailure)

# Define the disturbance distribution based on a normal distribution
px = Normal(0f0, 0.5f0)

# Construct the MDP
mdp = AdditiveAdversarialMDP(InvertedPendulumMDP(λcost=0, include_time_in_state=true), px)
S = state_space(mdp)
N = 100000

# construct the model 
QSA() = ContinuousNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
Pf() = ContinuousNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, 1, (x)-> -softplus(-(x-2)))))
A() = ContinuousNetwork(Chain(Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1, tanh), x -> 2f0 * x), 1)

G() = GaussianPolicy(A(), zeros(Float32, 1))

Protag() = ActorCritic(A(), DoubleNetwork(QSA(), QSA()))
Antag() = ActorCritic(G(), Pf())

AdvPol(p = Protag()) = AdversarialPolicy(p, Antag())

𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), S=S, N=50000, buffer_size=Int(1e5), buffer_init=1000)
π_td3 = solve(𝒮_td3, mdp)


# solve with IS
𝒮_isarl = ISARL_Continuous(π=AdvPol(), 
                           S=S, 
                           N=100000, 
                           N_IS_Samples=10,
                           N_recent_steps=1000,
                           π_explore=GaussianNoiseExplorationPolicy(Crux.LinearDecaySchedule(1., 0.0, floor(Int, 15000))),
                           ϵ_init = 1e-5,
                           px=px, 
                           buffer_size=Int(1e5), 
                           buffer_init=1000,)
π_isarl = solve(𝒮_isarl, mdp)

# check the number of failures in the buffers (compared to successs)
sum(𝒮_isarl.buffer[:fail])
sum(𝒮_isarl.buffer[:done])

# Plot the distribution of disturbances
histogram(𝒮_isarl.buffer[:x][:])

# Plot the value function in 2D
v(θ, t) = sum(value(antagonist(𝒮_isarl.π), [θ, -1, t, 0]))
heatmap(deg2rad(-25):0.01:deg2rad(25), 0:0.1:5, v)

# Show the distribution of data in the replay buffer
scatter(𝒮_isarl.buffer[:s][1, :], 𝒮_isarl.buffer[:s][3, :], marker_z = 𝒮_isarl.buffer[:done][1,:], xlabel="θ", ylabel="t", alpha=0.5)
vline!([deg2rad(20), deg2rad(-20)])



# Solve with DQN
𝒮_dqn = DQN(π=QS(as), S=S, N=N)
π_dqn = solve(𝒮_dqn, mdp)

pfail_dqn = Crux.failure(Sampler(mdp, π_dqn, S=S, max_steps=100), Neps=Int(1e3), threshold=100)
println("DQN Failure rate: ", pfail_dqn)

# solve with RARL
𝒮_rarl = RARL(π=AdvPol(), S=S, N=N)
π_rarl = solve(𝒮_rarl, mdp)

pfail_rarl = Crux.failure(Sampler(mdp, protagonist(π_rarl), S=S, max_steps=100), Neps=Int(1e5), threshold=100)
println("RARL Failure rate: ", pfail_rarl)


pfail_isarl = Crux.failure(Sampler(mdp, protagonist(π_isarl), S=S, max_steps=100), Neps=Int(1e5), threshold=100)
println("IS Failure rate: ", pfail_isarl)

pol = AdvPol()

𝒮_isarl.buffer


pol = AdversarialPolicy(π_dqn, Pf(xs), ϵGreedyPolicy(Crux.LinearDecaySchedule(1., 0.1, floor(Int, N/2)), xs))
𝒮_isarl = ISARL_Discrete(π=pol, S=S, N=N, xlogprobs=xlogprobs)
π_isarl = solve(𝒮_isarl, mdp)

