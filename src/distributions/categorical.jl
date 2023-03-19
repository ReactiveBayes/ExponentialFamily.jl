export Categorical
export lognormalizer, logpdf_sample_friendly

import Distributions: Categorical, probs

vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)

prod_analytical_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = ProdAnalyticalRuleAvailable()

convert_eltype(::Type{Categorical}, ::Type{T}, distribution::Categorical{R}) where {T <: Real, R <: Real} =
    Categorical(convert(AbstractVector{T}, probs(distribution)))

function Base.prod(::ProdAnalytical, left::Categorical, right::Categorical)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)
    return Categorical(mvec ./ norm)
end

probvec(dist::Categorical) = probs(dist)

function compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Categorical)
    return log(dot(probvec(left_dist), probvec(right_dist)))
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Categorical)
    logprobabilities = log.(probvec(dist))
    return ExponentialFamilyDistribution(Categorical, logprobabilities)
end

function Base.convert(::Type{Distribution}, η::ExponentialFamilyDistribution{Categorical})
    return Categorical(softmax(getnaturalparameters(η)))
end

check_valid_natural(::Type{<:Categorical}, params) = first(size(params)) >= 2

function lognormalizer(::ExponentialFamilyDistribution{Categorical})
    return 0.0
end

isproper(::ExponentialFamilyDistribution{Categorical}) = true

function logpdf_sample_friendly(dist::Categorical)
    p = probvec(dist)
    friendly = Categorical(p)
    return (friendly, friendly)
end

basemeasure(::Union{<:ExponentialFamilyDistribution{Categorical}, <:Categorical}, x) = 1.0
plus(::ExponentialFamilyDistribution{Categorical}, ::ExponentialFamilyDistribution{Categorical}) = Plus()
