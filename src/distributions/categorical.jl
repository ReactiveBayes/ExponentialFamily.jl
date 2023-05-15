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

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Categorical}, <:Categorical}, x) = 1.0
