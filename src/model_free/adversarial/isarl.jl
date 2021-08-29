function IS_DQN_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        𝒟[:done] .* Base.log.(𝒟[:fail] .+ 1f-16) .+ (1.f0 .- 𝒟[:done]) .* logsumexp(𝒫[:xlogprobs] .+ value(π, 𝒟[:sp]), dims=1)
        # 𝒟[:done] .* 𝒟[:fail] .+ (1.f0 .- 𝒟[:done]) .* sum(exp.(𝒫[:xlogprobs]) .* value(π, 𝒟[:sp]), dims=1)
end

function IS_estimate_log_pfail(π, px, s, Nsamples=10)
        xs_and_logpdfs = [exploration(π, s) for _ in 1:Nsamples]
        xs = [e[1] for e in xs_and_logpdfs]
        νlogpdfs = vcat([e[2] for e in xs_and_logpdfs]...)
        logpfails = vcat([value(π, s, x) for x in xs]...)
        logpxs = Float32.(vcat([logpdf.(px, x) for x in xs]...))

        -Float32(Base.log(Nsamples)) .+ logsumexp(νlogpdfs .+ logpfails .- logpxs, dims=1)
end

function estimate_log_pfail(π, px, s, Nsamples=10)
        xs = [reshape(rand(px, size(s,2)), :, size(s,2)) for _ in 1:Nsamples]
        logpfails = vcat([value(π, s, x) for x in xs]...)

        -Float32(Base.log(Nsamples)) .+ logsumexp(logpfails, dims=1)
end

function IS_Continuous_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        𝒟[:done] .* Base.log.(𝒟[:fail] .+ 1f-16) .+ (1.f0 .- 𝒟[:done]) .* estimate_log_pfail(π, 𝒫[:px], 𝒟[:sp], 𝒫[:N_IS_Samples])
end

function IS_L_KL(π, 𝒫, 𝒟; kwargs...)
        x, logνx = exploration(π, 𝒟[:s])
        logpfail_s = Zygote.ignore() do 
                logpfail_s = estimate_log_pfail(π, 𝒫[:px], 𝒟[:s], 𝒫[:N_IS_Samples])
                logpfail_s
        end
        logpx = Float32.(logpdf.(𝒫[:px], x))
        logpfail_sx = value(π, 𝒟[:s], x)
        
        -mean(exp.( logpx .+ logpfail_sx .- logνx .- logpfail_s) .* logνx)
end

function bce(π, 𝒫, 𝒟, y; loss=Flux.msle, weighted=false, name=:Qavg, info=Dict())
    Q = value(π, 𝒟[:s], 𝒟[:x]) 
    
    # Store useful information
    ignore() do
        info[name] = mean(Q)
    end
    
    Flux.Losses.binarycrossentropy(Q, y)
end

function x_mse(π, 𝒫, 𝒟, y; name=:Qavg, info=Dict())
    Q = value(π, 𝒟[:s], 𝒟[:x]) 
    
    # Store useful information
    ignore() do
        info[name] = mean(Q)
    end
    
    Flux.Losses.mse(Q, y)
end

function update_mixing_parameter(𝒟, 𝒫; info=Dict())
        ## This is for a per-step adversary
        indices = get_last_N_indices(𝒟, 𝒫.N_recent_steps)
        terminals = sum(𝒟[:done][:,indices])
        fails = sum(𝒟[:fail][:,indices])
        successes = terminals - fails
        r = (fails + 1) / (successes + fails + 2)
        info[:fail_ratio] = r
        
        rtarg = 𝒫.target_ratio
        𝒫.ϵx[1] = clamp(𝒫.ϵx[1] + 𝒫[:ϵ_lr]*(Base.log(r / (1-r)) - Base.log(rtarg / (1-rtarg))), -10, 10)
        info[:ϵx] = sigmoid(𝒫.ϵx[1])
        
        # this if for a per-episode adversary
        # terminals = sum(𝒟[:done])
        # fails = sum(𝒟[:fail])
        # successes = terminals - fails
        # r = (fails + 1) / (successes + fails + 2)
        # info[:fail_ratio] = r
        # if 𝒟[:done][1,end]
        #         # if the acceptance test passes then use the nominal policy, otherwise use the failure probability
        #         # if r is large there are lots of failures and the nominal policy is more likely to be used
        #         𝒫.ϵx[1] = rand() < r ? 1 : 0                
        # end
end

function ISARL_Continuous(;π::AdversarialPolicy, ΔN=50, N=1000, N_IS_Samples=10, N_recent_steps=100, px, ϵ_init = 0.5, target_ratio = 0.5, ϵ_lr=1e-3, π_explore=GaussianNoiseExplorationPolicy(LinearDecaySchedule(1., 0.0, floor(Int, N/2))), π_smooth::Policy=GaussianNoiseExplorationPolicy(0.1f0, ϵ_min=-0.5f0, ϵ_max=0.5f0), a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), x_a_opt::NamedTuple=(;), x_c_opt::NamedTuple=(;), log::NamedTuple=(;), 𝒫::NamedTuple=(;), kwargs...)
        𝒫 = (;π_smooth, N_IS_Samples, N_recent_steps, px, ϵ_lr, target_ratio, ϵx=[Base.log(ϵ_init / (1-ϵ_init))])
        π.A_explore = MixedPolicy((i)->sigmoid(𝒫.ϵx[1]), px) 
        AdversarialOffPolicySolver(;
                π=π, 
                log=LoggerParams(;dir="log/isarl", log...),
                ΔN=ΔN,
                N=N,
                𝒫 = 𝒫,
                a_opt = TrainingParams(;loss=TD3_actor_loss, name="actor_", a_opt...),
                x_a_opt = TrainingParams(;loss=IS_L_KL, name="actor_", x_a_opt...),
                c_opt = TrainingParams(;loss=double_Q_loss, name="critic_", epochs=ΔN, c_opt...),
                x_c_opt = TrainingParams(;loss=x_mse, name="critic_", epochs=ΔN, x_c_opt...),
                target_fn = TD3_target,
                required_columns=[:fail, :x],
                x_target_fn=IS_Continuous_target,
                π_explore=π_explore,
                post_experience_callback=update_mixing_parameter,
                kwargs...)
end

function ISARL_Discrete(;π::AdversarialPolicy, N::Int, ΔN=4, px, xlogprobs, N_recent_steps=1000, ϵ_init = 0.5, target_ratio = 0.5, ϵ_lr=1e-3, π_explore=ϵGreedyPolicy(LinearDecaySchedule(1., 0.1, floor(Int, N/2)), protagonist(π).outputs), c_opt::NamedTuple=(;),  x_c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...)
        𝒫 =(;px, xlogprobs, N_recent_steps, ϵ_lr, target_ratio, ϵx=[Base.log(ϵ_init / (1-ϵ_init))])
        π.A_explore = MixedPolicy((i)->sigmoid(𝒫.ϵx[1]), px) 
        
        ## This if for a per-episode adversary 
        # 𝒫 =(;px, xlogprobs, N_recent_episodes, ϵx=[ϵ_init])
        # π.A_explore = MixedPolicy((i)->𝒫.ϵx[1], px) 
        AdversarialOffPolicySolver(;
                π=π, 
                log=LoggerParams(;dir="log/isarl", log...),
                N=N,
                ΔN=ΔN,
                𝒫=𝒫,
                c_opt = TrainingParams(;loss=td_loss, name="critic_", epochs=ΔN, c_opt...),
                x_c_opt = TrainingParams(;loss=x_mse, name="critic_", epochs=ΔN, x_c_opt...),
                target_fn=DQN_target,
                required_columns=[:fail, :x, :xlogprob],
                x_target_fn=IS_DQN_target,
                post_experience_callback=update_mixing_parameter,
                π_explore=π_explore,
                # return_at_episode_end = true,
                kwargs...)
end

