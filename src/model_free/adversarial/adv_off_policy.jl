@with_kw mutable struct AdversarialOffPolicySolver <: Solver
    π # Policy
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(protagonist(π)) # Action space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 4 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of environment interactions
    param_optimizers::Dict{Any, TrainingParams} = Dict() # Training parameters for the parameters
    a_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the actor
    c_opt::TrainingParams # Training parameters for the critic
    x_param_optimizers::Dict{Any, TrainingParams} = Dict() # Training parameters for the parameters
    x_a_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the actor
    x_c_opt::TrainingParams # Training parameters for the critic
    𝒫::NamedTuple = (;) # Parameters of the algorithm
    desired_AP_ratio = 1 # Desired training ratio of protagonist to antagonist
    post_experience_callback = (buffer, 𝒫) -> nothing
    return_at_episode_end = false
    
    # Off-policy-specific parameters
    π⁻ = deepcopy(π)
    π_explore::Policy # exploration noise
    target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
    target_fn # Target for critic regression with input signature (π⁻, 𝒟, γ; i)
    x_target_fn # Target for critic regression with input signature (π⁻, 𝒟, γ; i)
    buffer_size = 1000 # Size of the buffer
    required_columns = Symbol[:x]
    buffer::ExperienceBuffer = ExperienceBuffer(S, A, buffer_size, required_columns) # The replay buffer
    buffer_init::Int = max(c_opt.batch_size, 200) # Number of observations to initialize the buffer with
    extra_buffers = [] # extra buffers (i.e. for experience replay in continual learning)
    buffer_fractions = [1.0] # Fraction of the minibatch devoted to each buffer
end

function POMDPs.solve(𝒮::AdversarialOffPolicySolver, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = buffer_like(𝒮.buffer, capacity=𝒮.c_opt.batch_size, device=device(𝒮.π))
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, S=𝒮.S, A=𝒮.A, max_steps=𝒮.max_steps, π_explore=𝒮.π_explore, required_columns=extra_columns(𝒮.buffer))
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)
    
    # Add performance metric for nominal performance
    s_nom = Sampler(mdp, protagonist(𝒮.π), S=𝒮.S, A=𝒮.A, max_steps=𝒮.max_steps, required_columns=setdiff(extra_columns(𝒮.buffer), [:x]))
    push!(𝒮.log.fns, (;kwargs...) -> Dict("nominal_undiscounted_return" => undiscounted_return(s_nom, Neps=10)))

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i)

    # Fill the buffer with initial observations before training
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i=𝒮.i, explore=true)
    
    N_antagonist = 1
    N_protagonist = 1
    
    antagonist_params = (antagonist(𝒮.π), antagonist(𝒮.π⁻), 𝒮.x_target_fn, 𝒮.x_param_optimizers, 𝒮.x_a_opt, 𝒮.x_c_opt, "x_")
    protagonist_params = (protagonist(𝒮.π), protagonist(𝒮.π⁻), 𝒮.target_fn, 𝒮.param_optimizers, 𝒮.a_opt, 𝒮.c_opt, "")
    
    # Loop over the desired number of environment interactions
    # for 𝒮.i in range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
    stop =𝒮.i + 𝒮.N - 𝒮.ΔN
    while 𝒮.i <= stop
        # Sample transitions into the replay buffer
        D = steps!(s, Nsteps=𝒮.ΔN, explore=true, i=𝒮.i, return_at_episode_end=𝒮.return_at_episode_end)
        push!(𝒮.buffer, D)
        𝒮.i += length(D[:done][:])
        
        # callback for potentially updating the buffer
        info = Dict()
        𝒮.post_experience_callback(𝒮.buffer, 𝒮.𝒫; info=info)
        
        infos = [info]
        
        train_over = []
        curr_ratio = N_antagonist / N_protagonist
        if curr_ratio <= 𝒮.desired_AP_ratio
            push!(train_over, antagonist_params)
            N_antagonist += 1
        end
        if curr_ratio >= 𝒮.desired_AP_ratio
            push!(train_over, protagonist_params)
            N_protagonist += 1
        end
        
        ## Loop over the antagonist and protagonist
        for (π, π⁻, target_fn, param_optimizers, a_opt, c_opt, prefix) in train_over
        
            # Loop over the desired number of training steps
            for epoch in 1:c_opt.epochs
                # Sample a batch
                rand!(𝒟, 𝒮.buffer, 𝒮.extra_buffers..., fracs=𝒮.buffer_fractions, i=𝒮.i)
                
                # initialize the info
                info = Dict()
                
                # Compute the target
                y = target_fn(π⁻, 𝒮.𝒫, 𝒟, γ, i=𝒮.i)
                
                # Train parameters
                for (θs, p_opt) in param_optimizers
                    train!(θs, (;kwargs...) -> p_opt.loss(π, 𝒮.𝒫, 𝒟; kwargs...), p_opt, info=info)
                end
                
                # Train the critic
                if ((epoch-1) % c_opt.update_every) == 0
                    train!(critic(π), (;kwargs...) -> c_opt.loss(π, 𝒮.𝒫, 𝒟, y; kwargs...), c_opt, info=info)
                end
                
                # Train the actor 
                if !isnothing(a_opt) && ((epoch-1) % a_opt.update_every) == 0
                    train!(actor(π), (;kwargs...) -> a_opt.loss(π, 𝒮.𝒫, 𝒟; kwargs...), a_opt, info=info)
                
                    # Update the target network
                    𝒮.target_update(π⁻, π)
                end
                
                # Store the training information
                push!(infos, Dict(string(prefix, k) => v for (k,v) in info))
                
            end
            # If not using a separate actor, update target networks after critic training
            isnothing(a_opt) && 𝒮.target_update(π⁻, π, i=𝒮.i + 1:𝒮.i + 𝒮.ΔN)
            
        end # end loop over A and P
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, aggregate_info(infos))
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end

