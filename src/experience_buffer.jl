const MinHeap = MutableBinaryHeap{Float32, DataStructures.FasterForward}

# Efficient inverse query for fenwick tree : adapted from https://codeforces.com/blog/entry/61364
function inverse_query(t::FenwickTree, v, N = length(t))
    tot, pos = 0, 0
    for i=floor(Int, log2(N)):-1:0
        new_pos = pos + 1 << i
        if new_pos <= N && tot + t.bi_tree[new_pos] < v
            tot += t.bi_tree[new_pos]
            pos = new_pos
        end
    end
    pos + 1
end

Base.getindex(t::FenwickTree, i::Int) = prefixsum(t, i) - prefixsum(t, i-1)

DataStructures.update!(t::FenwickTree, i, v) = inc!(t, i, v - t[i])

Base.zero(s::Type{Symbol}) = :zero # For initializing symbolic arrays

# construction of common mdp data
function mdp_data(S::T1, A::T2, capacity::Int, extras::Array{Symbol} = Symbol[]; ArrayType = Array, R = Float32, D = Bool, W = Float32) where {T1 <: AbstractSpace, T2 <: AbstractSpace}
    data = Dict{Symbol, ArrayType}(
        :s => ArrayType(fill(zero(type(S)), dim(S)..., capacity)), 
        :a => ArrayType(fill(zero(type(A)), dim(A)..., capacity)), 
        :sp => ArrayType(fill(zero(type(S)), dim(S)..., capacity)), 
        :r => ArrayType(fill(zero(R), 1, capacity)), 
        :done => ArrayType(fill(zero(D), 1, capacity))
        )
    for k in extras
        if k in [:return, :logprob, :xlogprob, :advantage, :cost]
            data[k] = ArrayType(fill(zero(R), 1, capacity))
        elseif k in [:weight]
            data[k] = ArrayType(fill(one(R), 1, capacity))
        elseif k in [:episode_end, :fail]
            data[k] = ArrayType(fill(false, 1, capacity))
        elseif k in [:t, :i]
            data[k] = ArrayType(fill(0, 1, capacity))
        elseif k in [:s0]
            data[k] = ArrayType(fill(zero(type(S)), dim(S)..., capacity))
        elseif k in [:x] # Disturbances for adversarial policy
            data[k] = ArrayType(fill(zero(type(A)), dim(A)..., capacity))
        else
            CRUX_WARNINGS && @warn "Unrecognized key: $k"
        end
    end
    data
end

## Prioritized replay parameters
@with_kw mutable struct PriorityParams
    minsort_priorities::MinHeap
    priorities::FenwickTree
    α::Float32 = 0.6
    β::Function = (i) -> 0.5f0
    max_priority::Float32 = 1.0
end

PriorityParams(N::Int; kwargs...) = PriorityParams(;minsort_priorities=MinHeap(fill(Inf32, N)), priorities=FenwickTree(fill(0f0, N)), kwargs...)
PriorityParams(N::Int, pp::PriorityParams) = PriorityParams(N, α=pp.α, β=pp.β, max_priority=pp.max_priority)
    
## Experience Buffer stuff
@with_kw mutable struct ExperienceBuffer{T <: AbstractArray} 
    data::Dict{Symbol, T}
    elements::Int64
    next_ind::Int64
    indices::Vector{Int}
    priority_params::Union{Nothing, PriorityParams}
end

function ExperienceBuffer(data::Dict{Symbol, T}; elements=capacity(data), next_ind=1, indices=[], prioritized::Bool=false, priority_params::NamedTuple=(;)) where {T <: AbstractArray}
    pp = nothing
    if prioritized
        !haskey(data, :weight) && (data[:weight] = device(data)((ones(Float32, 1, capacity(data)))))
        pp = PriorityParams(capacity(data); priority_params...)
    end
    ExperienceBuffer(data, elements, next_ind, indices, pp)
end

function ExperienceBuffer(S::T1, A::T2, capacity::Int, extras::Array{Symbol}=Symbol[]; device=cpu, 
                          prioritized=false, priority_params::NamedTuple=(;), R=Float32, D=Bool, W=Float32) where {T1 <: AbstractSpace, T2 <: AbstractSpace}
    Atype = device == gpu ? CuArray : Array
    data = mdp_data(S, A, capacity, extras, ArrayType=Atype, R=R, D=D, W=W)
    ExperienceBuffer(data, elements=0, prioritized=prioritized, priority_params=priority_params)
end

function buffer_like(b::ExperienceBuffer; capacity=capacity(b), device=device(b))
    data = Dict(k=>device(Array{eltype(v)}(undef, size(v)[1:end-1]..., capacity)) for (k,v) in b.data)
    ExperienceBuffer(data, 0, 1, Int[], isprioritized(b) ? PriorityParams(capacity, b.priority_params) : nothing)
end

function Flux.gpu(b::ExperienceBuffer)
    data = Dict{Symbol, CuArray}(k => v |> gpu for (k,v) in b.data)
    ExperienceBuffer(data, b.elements, b.next_ind, b.indices, b.priority_params)
end

function Flux.cpu(b::ExperienceBuffer)
    data = Dict{Symbol, Array}(k => v |> cpu for (k,v) in b.data)
    ExperienceBuffer(data, b.elements, b.next_ind, b.indices, b.priority_params)
end

function clear!(b::ExperienceBuffer)
    b.elements = 0
    b.next_ind = 1
    b.indices = Int[]
    isprioritized(b) && (b.priority_params = PriorityParams(capacity(b), b.priority_params))
    b
end 

function Base.hcat(buffers::ExperienceBuffer...; kwargs...)
    data = Dict(k=>typeof(v)(undef, size(v)[1:end-1]...,0) for (k,v) in buffers[1].data)
    for b in buffers[1:end]
        @assert keys(data) == keys(b)
        for k in keys(b)
            data[k] = cat(data[k], b[k], dims=ndims(b[k]))
        end
    end
    ExperienceBuffer(data; kwargs...)
end

function Random.shuffle!(b::ExperienceBuffer)
    new_i = shuffle(1:length(b))
    for k in keys(b)
        b[k] .= bslice(b[k], new_i)
    end
    b
end

function split_batches(N, fracs)
    @assert sum(fracs) ≈ 1
    batches = floor.(Int, N .* fracs)
    batches[1] += N - sum(batches)
    batches
end

function Base.split(b::ExperienceBuffer, fracs)
    buffers = ExperienceBuffer[]
    start = 1
    for batch in split_batches(length(b), fracs)
        push!(buffers, ExperienceBuffer(minibatch(b, start:start+batch-1)))
        start += batch
    end
    buffers
end

function normalize!(b::ExperienceBuffer, S::AbstractSpace, A::AbstractSpace)
    haskey(b, :s) && (b[:s] .= tovec(b[:s], S))
    haskey(b, :sp) && (b[:sp] .= tovec(b[:sp], S))
    A isa ContinuousSpace && (b[:a] .= tovec(b[:a], A))
    b
end

function trim!(b::ExperienceBuffer, range)
    for k in keys(b)
        b.data[k] = bslice(b.data[k], range)
    end
    b.elements = min(b.elements, length(range))
    if b.next_ind > b.elements
        b.next_ind = 1
    end
    b
end


minibatch(b::ExperienceBuffer, indices) = Dict(k => bslice(b.data[k], indices) for k in keys(b))

Base.getindex(b::ExperienceBuffer, key::Symbol) = bslice(b.data[key], 1:b.elements)

Base.keys(b::ExperienceBuffer) = keys(b.data)

extra_columns(b::ExperienceBuffer) = collect(setdiff(keys(b), [:s, :a, :sp, :r, :done]))

Base.first(b::ExperienceBuffer) = first(b.data)

Base.haskey(b::ExperienceBuffer, k) = haskey(b.data, k)

Base.length(b::ExperienceBuffer) = b.elements

DataStructures.capacity(b::Union{Dict{Symbol, T}, ExperienceBuffer}) where {T <: AbstractArray} = size(first(b)[2])[end]

isprioritized(b::ExperienceBuffer) = !isnothing(b.priority_params)

device(b::ExperienceBuffer{CuArray}) = gpu
device(b::Dict{Symbol, CuArray}) = gpu
device(b::ExperienceBuffer{Array}) = cpu
device(b::Dict{Symbol, Array}) = cpu

function episodes(b::ExperienceBuffer)
    if haskey(b, :episode_end)
        ep_ends = findall(b[:episode_end][1,:])
        ep_starts = [1, ep_ends[1:end-1] .+ 1 ...]
    elseif haskey(b, :t)
        ep_starts = findall(b[:t][1,:] .== 1)
        ep_ends = [ep_starts[2:end] .- 1 ..., length(b)]
    else
        error("Need :episode_end flag or :t column to determine episodes")
    end
    zip(ep_starts, ep_ends)
end

function get_last_N_indices(b::ExperienceBuffer, N)
    # Make sure we don't exceed the number of elements actually stored in the buffer
    N = min(length(b), N)
    C = capacity(b)
    start_index = mod1(b.next_ind - N, C)
    mod1.(start_index:start_index + N - 1, C)
end

# Note: data can be a dictionary or an experience buffer
function Base.push!(b::ExperienceBuffer, data; ids = nothing)
    ids = isnothing(ids) ? UnitRange(1, size(data[first(keys(data))], 2)) : ids
    N, C = length(ids), capacity(b)
    I = mod1.(b.next_ind:b.next_ind + N - 1, C)
    for k in keys(b)
        if !haskey(data, k)
            # CRUX_WARNINGS && @warn "Pushed data does not contain $k"
            continue
        end
        v1 = bslice(b.data[k], I)
        v2 = collect(bslice(data[k], ids))
        @assert size(v1)[1:end-1] == size(v2)[1:end-1]
        copyto!(v1, v2)
    end
    isprioritized(b) && update_priorities!(b, I, b.priority_params.max_priority*ones(N))
        
    b.elements = min(C, b.elements + N)
    b.next_ind = mod1(b.next_ind + N, C)
    I
end

function update_priorities!(b, I::AbstractArray, v::AbstractArray)
    for i = 1:length(I)
        val = v[i] + eps(Float32)
        update!(b.priority_params.priorities, I[i], val^b.priority_params.α)
        update!(b.priority_params.minsort_priorities, I[i], val^b.priority_params.α)
        b.priority_params.max_priority = max(val, b.priority_params.max_priority)
    end
end

function Random.rand!(target::ExperienceBuffer, source::ExperienceBuffer...; i = 1, fracs = ones(length(source))./length(source))
    batches = split_batches(capacity(target), fracs)
    for (b, B) in zip(source, batches)
        B == 0 && continue
        isprioritized(b) ? prioritized_sample!(target, b, i=i, B=B) : uniform_sample!(target, b, B=B)
    end
end

function uniform_sample!(target::ExperienceBuffer, source::ExperienceBuffer; B = capacity(target))
    ids = rand(1:length(source), B)
    target.indices = ids
    push!(target, source, ids = ids)
end

# With guidance from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
function prioritized_sample!(target::ExperienceBuffer, source::ExperienceBuffer; i = 1, B = capacity(target))
    @assert haskey(source, :weight) 
    N = length(source)
    prs = source.priority_params.priorities
    ptot = prefixsum(prs, N)
    Δp = ptot / B
    target.indices = [inverse_query(prs, (j + rand() - 1) * Δp, N-1) for j=1:B]
    pmin = first(source.priority_params.minsort_priorities) / ptot
    max_w = (pmin*N)^(-source.priority_params.β(i))
    source[:weight][1, target.indices] .= [(N * prs[id] / ptot)^source.priority_params.β(i) for id in target.indices] ./ max_w
    
    push!(target, source, ids = target.indices)
end


