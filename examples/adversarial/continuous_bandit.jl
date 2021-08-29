using POMDPs, Crux, Flux, POMDPGym, Random, Distributions, POMDPPolicies
Crux.set_function("isfailure", POMDPGym.isfailure)

# Define the disturbance distribution based on a normal distribution
px = Normal(0f0, 0.5f0)

# Construct the MDP
mdp = AdditiveAdversarialMDP(ContinuousBanditMDP(), px)
S = state_space(mdp)
N = 10000

# construct the model 
QSA() = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
Pf() = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1, (x)-> -softplus(-(x-2)))))
A() = ContinuousNetwork(Chain(Dense(1, 64, relu), Dense(64, 64, relu), Dense(64, 1, tanh), x -> 2f0 * x), 1)

G() = GaussianPolicy(A(), zeros(Float32, 1))

Protag() = ActorCritic(A(), DoubleNetwork(QSA(), QSA()))
Antag() = ActorCritic(G(), Pf())

AdvPol(p = Protag()) = AdversarialPolicy(p, Antag())

𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), S=S, N=50000, buffer_size=Int(1e5), buffer_init=1000)
π_td3 = solve(𝒮_td3, mdp)


# solve with IS
𝒮_isarl = ISARL_Continuous(π=AdvPol(𝒮_td3.π), 
                           S=S, 
                           N=50000, 
                           N_IS_Samples=100,
                           N_recent_steps=1000,
                           π_explore=GaussianNoiseExplorationPolicy(Crux.LinearDecaySchedule(0., 0.0, floor(Int, 15000))),
                           ϵ_init = 1e-5,
                           px=px, 
                           buffer_size=Int(1e5), 
                           buffer_init=1000,)
π_isarl = solve(𝒮_isarl, mdp)


plot(-3:0.1:3, [exp(value(antagonist(𝒮_isarl.π), [0], [i])[1]) for i=-3:0.1:3], label="Probability of failure")

plot!(-3:0.1:3, [exp(logpdf(antagonist(𝒮_isarl.π).A, [0], [i])[1]) for i=-3:0.1:3], label="Failure Policy")

plot!(-3:0.1:3, [exp(logpdf(px, i)) for i=-3:0.1:3], label="Px")

