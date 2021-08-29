function GAIL_D_loss(gan_loss)
    (D, 𝒫, 𝒟_ex, 𝒟_π; info = Dict()) ->begin
        Lᴰ(gan_loss, D, 𝒟_ex[:a], 𝒟_π[:a], yD = (𝒟_ex[:s],), yG = (𝒟_π[:s],))
    end
end
    

function OnPolicyGAIL(;π, S, A=action_space(π), 𝒟_demo, normalize_demo::Bool=true, D::ContinuousNetwork, solver=PPO, gan_loss::GANLoss=GAN_BCELoss(), d_opt::NamedTuple=(;), log::NamedTuple=(;),  kwargs...)
    d_opt = TrainingParams(;loss = GAIL_D_loss(gan_loss), name="discriminator_", d_opt...)
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, A))
    𝒟_demo = 𝒟_demo |> device(π)
    
    function GAIL_callback(𝒟; info=Dict())
        batch_train!(D, d_opt, (;), 𝒟_demo, 𝒟, info=info)
        
        discriminator_signal = haskey(𝒟, :advantage) ? :advantage : :return
        D_out = value(D, 𝒟[:a], 𝒟[:s])
        r = Base.log.(sigmoid.(D_out) .+ 1f-5) .- Base.log.(1f0 .- sigmoid.(D_out) .+ 1f-5)
        𝒟[discriminator_signal] .= r # This is swapped because a->x and s->y and the convention for GANs is D(x,y)
    end
    𝒮 = solver(;π=π, S=S, A=A, post_batch_callback=GAIL_callback, log=(dir="log/onpolicygail", period=500, log...), kwargs...)
    𝒮.c_opt = nothing # disable the critic 
    𝒮
end

function OffPolicyGAIL(;π, S, A=action_space(π), 𝒟_demo, failure_classifier = nothing, normalize_demo::Bool=true, D::ContinuousNetwork, solver=SAC, d_opt::NamedTuple=(epochs=5,), log::NamedTuple=(;), NDA_GAIL_λ_opt::NamedTuple=(;), λ = 10f0, λtarget=30f-3, kwargs...)
    d_opt = TrainingParams(;name="discriminator_", loss=()->nothing, d_opt...)
    
    accel = 1:3
    goal = 4:19
    gyro = 20:22 
    hazards = 23:38
    magnetometer = 39:41
    vases = 42:57 
    velocimeter = 58:60
    
    indices = [hazards...]
    
    dev = device(π)
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, A))
    𝒟_demo = 𝒟_demo |> dev
    
    # function NDA_GAIL_λ_loss(π, 𝒫, 𝒟; info = Dict())
    #     val = ignore() do
    #         info["lambda"] = softplus.(𝒫[:logλ][1])
    # 
    #         isfailure = sigmoid.(failure_classifier(𝒟[:s][indices, :]))
    #         # isfailure = 𝒟[:cost]
    #         mean(isfailure .- 𝒫[:λtarget])
    # 
    #         # mean(((Base.log.(isfailure .+ 1f-5) .- Base.log.(1f0 .- isfailure .+ 1f-5)) .- 
    #               # (Base.log.(𝒫[:λtarget] .+ 1f-5) .- Base.log.(1f0 .- 𝒫[:λtarget] .+ 1f-5))))
    #     end
    # 
    #     -softplus.(𝒫[:logλ][1]) * val
    # end
    # 
    # 𝒫 = (;logλ = [λ], λtarget)
    𝒮 = solver(;π=π, S=S, A=A, 
            post_experience_callback=(𝒟; kwargs...) -> 𝒟[:r] .= 0, 
            log=(dir="log/offpolicygail", period=500, log...),
            # 𝒫 = 𝒫,
            # param_optimizers = isnothing(failure_classifier) ? Dict() : Dict(Flux.params(𝒫[:logλ]) => TrainingParams(;loss=NDA_GAIL_λ_loss, name="NDA_GAIL_lambda_", batch_size=1024, NDA_GAIL_λ_opt...)),
            kwargs...)
    
    B = d_opt.batch_size
    𝒟_π_batch = buffer_like(𝒮.buffer, capacity=B, device=dev)
    𝒟_demo_batch = deepcopy(𝒟_π_batch)
    𝒟_demo_π_batch = deepcopy(𝒟_π_batch)
    
    function GAIL_callback(𝒟; info=Dict())
        for i=1:d_opt.epochs
            # Sample minibatchs
            rand!(𝒟_demo_batch, 𝒟_demo)
            # rand!(𝒟_demo_π_batch, 𝒟_demo)
            # 𝒟_demo_π_batch.data[:a] = action(π, 𝒟_demo_π_batch[:s])
            rand!(𝒟_π_batch, 𝒮.buffer)
            
            𝒟_full = hcat(𝒟_demo_batch, #=𝒟_demo_π_batch, =# 𝒟_π_batch)
            
            # concat inputs
            x = cat(𝒟_full[:s], 𝒟_full[:a], dims=1)
            
            # Add some noise
            # x .+= Float32.(rand(Normal(0, 0.2f0), size(x))) |> dev
            
            # Create labels
            y_demo = ones(Float32, 1, B)
            # y_demo_π = zeros(Float32, 1, B)
            y_π = zeros(Float32, 1, B)
            y = cat(y_demo, #=y_demo_π,=# y_π, dims=2) |> dev
            
            train!(D, (;kwargs...) -> Flux.Losses.logitbinarycrossentropy(D(x), y), d_opt, info=info)
        end
        # # replace the bufffer
        rand!(𝒟_demo_batch, 𝒟_demo)
        rand!(𝒟_demo_π_batch, 𝒟_demo)
        𝒟_demo_π_batch.data[:a] = action(π, 𝒟_demo_π_batch[:s])
        rand!(𝒟_π_batch, 𝒮.buffer)
        
        𝒟_full = hcat(𝒟_demo_batch, 𝒟_demo_π_batch, 𝒟_π_batch)
        
        for k in keys(𝒟)
            𝒟.data[k] = 𝒟_full.data[k]
        end
        𝒟.elements = 𝒟_full.elements
        𝒟.next_ind = 𝒟_full.next_ind
        
        ## Compute the rewards
        D_out = Flux.sigmoid.(value(D, 𝒟[:s], 𝒟[:a]))
        # 𝒟[:r] .= Base.log.(D_out .+ 1f-5) .- Base.log.(1f0 .- D_out .+ 1f-5)
        𝒟[:r] .= -Base.log.(1f0 .- D_out .+ 1f-5)
        if !isnothing(failure_classifier)
            # isfailure = sigmoid.( failure_classifier(𝒟[:s][indices, :]))
            isfailure = 𝒟[:cost]
            𝒟[:r] .-=  10f0 .* isfailure
        end
        
        # 𝒟[:r] = D_out .- 0.5f0*isfailure
        
        
        # 𝒟[:r] .-= exp(𝒫[:logλ][1]) * ((Base.log.(isfailure .+ 1f-5) .- Base.log.(1f0 .- isfailure .+ 1f-5)) .- 
        #       (Base.log.(𝒫[:λtarget] .+ 1f-5) .- Base.log.(1f0 .- 𝒫[:λtarget] .+ 1f-5)))
        # end
                  
    end
    
    𝒮.post_batch_callback = GAIL_callback
    𝒮
end

(x) ->nothing


# Multi-headed gail
# function OffPolicyGAIL(;π, S, A=action_space(π), 𝒟_demo, 𝒟_ndas::Array{ExperienceBuffer} = ExperienceBuffer[], normalize_demo::Bool=true, D::ContinuousNetwork, solver=SAC, d_opt::NamedTuple=(epochs=5,), log::NamedTuple=(;), kwargs...)
#     d_opt = TrainingParams(;name="discriminator_", loss=()->nothing, d_opt...)
# 
#     dev = device(π)
#     normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, A))
#     𝒟_demo = 𝒟_demo |> dev
#     N_nda = length(𝒟_ndas)
#     λ_nda = 0.5f0*Float32(-1 / N_nda)
#     N_datasets = 2 + N_nda
#     for i in 1:N_nda
#         𝒟_ndas[i] = normalize_demo && normalize!(deepcopy(𝒟_ndas[i]), S, A)
#         𝒟_ndas[i] = 𝒟_ndas[i] |> dev
#     end
# 
# 
#     𝒮 = solver(;π=π, S=S, A=A, 
#             post_experience_callback=(𝒟; kwargs...) -> 𝒟[:r] .= 0, 
#             log=(dir="log/offpolicygail", period=500, log...),
#             kwargs...)
# 
# 
# 
#     B = d_opt.batch_size
#     𝒟_batch = buffer_like(𝒮.buffer, capacity=B, device=dev)
# 
#     𝒟_demo_batch = deepcopy(𝒟_batch)
#     𝒟_demo_π_batch = deepcopy(𝒟_batch)
#     𝒟_ndas_batch = [deepcopy(𝒟_batch) for 𝒟_nda in 𝒟_ndas]
#     𝒟_ndas_π_batch = [deepcopy(𝒟_batch) for 𝒟_nda in 𝒟_ndas]
# 
#     function GAIL_callback(𝒟; info=Dict())
#         for i=1:d_opt.epochs
#             # Sample minibatchs
#             rand!(𝒟_demo_batch, 𝒟_demo)
#             # rand!(𝒟_demo_π_batch, 𝒟_demo)
#             # 𝒟_demo_π_batch.data[:a] = action(π, 𝒟_demo_π_batch[:s])
#             rand!(𝒟_batch, 𝒮.buffer)
#             for i in 1:N_nda
#                 rand!(𝒟_ndas_batch[i], 𝒟_ndas[i])
#                 # rand!(𝒟_ndas_π_batch[i], 𝒟_ndas[i])
#                 # 𝒟_ndas_π_batch[i].data[:a] = action(π, 𝒟_ndas_π_batch[i][:s])
#             end
# 
#             𝒟_full = hcat(𝒟_demo_batch, 𝒟_batch, 𝒟_ndas_batch...)
# 
#             # concat inputs
#             x = cat(𝒟_full[:s], 𝒟_full[:a], dims=1)
# 
#             # Add some noise
#             x .+= Float32.(rand(Normal(0, 0.2f0), size(x))) |> dev
# 
#             # Create labels
#             y_demo = Flux.onehotbatch(ones(Int, B), 1:N_datasets)
#             # y_demo_π = Flux.onehotbatch(2*ones(Int, B), 1:N_datasets)
#             y_π = Flux.onehotbatch(2*ones(Int, B), 1:N_datasets)
#             y_ndas = [Flux.onehotbatch((i+2)*ones(Int, B), 1:N_datasets) for i=1:N_nda]
#             # y_ndas_π = [Flux.onehotbatch(2*ones(Int, B), 1:N_datasets) for i=1:N_nda]
# 
#             y = cat(y_demo, y_π, y_ndas..., dims=2) |> dev
# 
# 
#             # println(gradient_penalty(D, x_demo, x_π))
#             # + 10f0 * gradient_penalty(D, x_demo, x_π)
#             train!(D, (;kwargs...) -> Flux.Losses.logitcrossentropy(D(x), y), d_opt, info=info)
#         end
# 
#         ## replace the bufffer
#         # rand!(𝒟_demo_batch, 𝒟_demo)
#         # rand!(𝒟_demo_π_batch, 𝒟_demo)
#         # 𝒟_demo_π_batch.data[:a] = action(π, 𝒟_demo_π_batch[:s])
#         # rand!(𝒟_batch, 𝒮.buffer)
#         # for i in 1:N_nda
#         #     rand!(𝒟_ndas_batch[i], 𝒟_ndas[i])
#         #     rand!(𝒟_ndas_π_batch[i], 𝒟_ndas[i])
#         #     𝒟_ndas_π_batch[i].data[:a] = action(π, 𝒟_ndas_π_batch[i][:s])
#         # end
#         # 
#         # 𝒟_full = hcat(𝒟_demo_batch, 𝒟_batch, 𝒟_ndas_batch...)
#         # 
#         # for k in keys(𝒟)
#         #     𝒟.data[k] = 𝒟_full.data[k]
#         # end
#         # 𝒟.elements = 𝒟_full.elements
#         # 𝒟.next_ind = 𝒟_full.next_ind
# 
#         ## Compute the rewards
#         D_out = Flux.softmax(value(D, 𝒟[:s], 𝒟[:a]))
#         # we = max.(D_out[1, :], 1f0 .- D_out[1, :])
#         # w_ndas = [-D_out[i, :] .* (1f0 .- we) ./ sum(D_out[3:N_datasets, :], dims=1)[:] for i=3:N_datasets]
#         # w = cat(we, zeros(Float32, size(we)), w_ndas..., dims=2)'
#         w = [1f0, 0f0, λ_nda*ones(Float32, N_nda)...] |> dev
# 
#         𝒟[:r] .= sum((Base.log.(D_out .+ 1f-5) .- Base.log.(1f0 .- D_out .+ 1f-5)) .* w, dims=1)
#     end
# 
#     𝒮.post_batch_callback = GAIL_callback
#     𝒮
# end

