export Categorical
export logpartition, logpdf_sample_friendly

import Distributions: Categorical, probs

vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)

prod_analytical_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = ClosedProd()

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
    logprobabilities = log.(probvec(dist))
    return KnownExponentialFamilyDistribution(Categorical, logprobabilities)
end

function Base.convert(::Type{Distribution}, η::KnownExponentialFamilyDistribution{Categorical})
    return Categorical(softmax(getnaturalparameters(η)))
end

check_valid_natural(::Type{<:Categorical}, params) = first(size(params)) >= 2

function logpartition(::KnownExponentialFamilyDistribution{Categorical})
    return 0.0
end

isproper(::KnownExponentialFamilyDistribution{Categorical}) = true

function logpdf_sample_friendly(dist::Categorical)
    p = probvec(dist)
    friendly = Categorical(p)
    return (friendly, friendly)
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Categorical}, <:Categorical}, x) = 1.0
