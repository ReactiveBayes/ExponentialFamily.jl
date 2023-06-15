export Categorical
export logpartition

import Distributions: Categorical, probs

vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)

prod_closed_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = ClosedProd()

convert_eltype(::Type{Categorical}, ::Type{T}, distribution::Categorical{R}) where {T <: Real, R <: Real} =
    Categorical(convert(AbstractVector{T}, probs(distribution)))

function Base.prod(::ClosedProd, left::Categorical, right::Categorical)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)
    return Categorical(mvec ./ norm)
end

probvec(dist::Categorical) = probs(dist)

function compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Categorical)
    return log(dot(probvec(left_dist), probvec(right_dist)))
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Categorical)
    p = probvec(dist)
    η = log.(p / p[end])
    return KnownExponentialFamilyDistribution(Categorical, η)
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Categorical})
    η = getnaturalparameters(exponentialfamily)
    return Categorical(softmax(η))
end

check_valid_natural(::Type{<:Categorical}, params) = first(size(params)) >= 2

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Categorical})
    η = getnaturalparameters(exponentialfamily)
    return log(sum(exp.(η)))
end

isproper(::KnownExponentialFamilyDistribution{Categorical}) = true


function fisherinformation(expfamily::KnownExponentialFamilyDistribution{Categorical})
    η = getnaturalparameters(expfamily)
    I = Matrix{Float64}(undef, length(η), length(η))
    @inbounds for i in 1:length(η)
        I[i, i] = exp(η[i]) * (sum(exp.(η)) - exp(η[i])) / (sum(exp.(η)))^2
        for j in 1:i-1
            I[i, j] = -exp(η[i]) * exp(η[j]) / (sum(exp.(η)))^2
            I[j, i] = I[i, j]
        end
    end
    return I
end

function fisherinformation(dist::Categorical)
    p = probvec(dist)
    I = Matrix{Float64}(undef, length(p), length(p))
    @inbounds for i in 1:length(p)
        I[i, i] = 1 / p[i]
        for j in 1:i-1
            I[i, j] = 0
            I[j, i] = I[i, j]
        end
    end
    return I
end

function basemeasure(::Union{<:KnownExponentialFamilyDistribution{Categorical}, <:Categorical}, x::Real) 
    @assert typeof(x) <: Integer "Categorical should be evaluated at integer values"
    return 1.0
end

function basemeasure(::Union{<:KnownExponentialFamilyDistribution{Categorical}, <:Categorical}, x::Vector) 
    @assert typeof(x) <: Vector{<:Integer} "One-hot coded Categorical should be evaluated at integer values"
    @assert sum(x) == 1 "One-hot coded Categorical should sum to 1"
    return 1.0
end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Categorical}, x::Real) 
    @assert typeof(x) <: Integer "Categorical should be evaluated at integer values"
    K = length(getnaturalparameters(ef))
    @assert x <= K "Categorical distribution should be evaluated at values that are leq than the size of $(ef)"
    ss = zeros(K)
    [ss[k] = 1 for k=1:K if x==k]
    return ss
end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Categorical}, x::Vector) 
    @assert typeof(x) <: Vector{<:Integer} "One-hot coded Categorical should be evaluated at integer values"
    @assert sum(x) == 1 "One-hot coded Categorical should sum to 1"
    K = length(getnaturalparameters(ef))
    @assert K == length(x) "x should be same length as $(ef)"
    ss = zeros(K)
    [ss[k] = 1 for k=1:K if x[k]==1]
    return ss
end


function sufficientstatistics(dist::Categorical, x::Real) 
    @assert typeof(x) <: Integer "Categorical should be evaluated at integer values"
    K = length(probvec(dist))
    @assert x <= K "Categorical distribution should be evaluated at values that are leq than the size of $(ef)"
    ss = zeros(K)
    [ss[k] = 1 for k=1:K if x==k]
    return ss
end

function sufficientstatistics(dist::Categorical, x::Vector) 
    @assert typeof(x) <: Vector{Integer} "One-hot coded Categorical should be evaluated at integer values"
    @assert sum(x) == 1 "One-hot coded Categorical should sum to 1"
    K = length(probvec(dist))
    @assert K == length(x) "x should be same length as $(ef)"
    ss = zeros(K)
    [ss[k] = 1 for k=1:K if x[k]==1]
    return ss
end