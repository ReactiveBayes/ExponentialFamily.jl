export Bernoulli

import Distributions: Bernoulli, succprob, failprob, logpdf
import StatsFuns: logistic, logit

vague(::Type{<:Bernoulli}) = Bernoulli(0.5)
probvec(dist::Bernoulli) = (failprob(dist), succprob(dist))

include("./product.jl")
include("./conversions.jl")

function compute_logscale(new_dist::Bernoulli, left_dist::Bernoulli, right_dist::Bernoulli)
    left_p = succprob(left_dist)
    right_p = succprob(right_dist)
    a = left_p * right_p + (one(left_p) - left_p) * (one(right_p) - right_p)
    return log(a)
end

function compute_logscale(new_dist::Categorical, left_dist::Bernoulli, right_dist::Categorical)
    p_left = probvec(left_dist)
    p_right = probvec(right_dist)

    Z = if length(p_left) >= length(p_right)
        dot(p_left, vcat(p_right..., zeros(length(p_left) - length(p_right))))
    else
        dot(p_right, vcat(p_left..., zeros(length(p_right) - length(p_left))))
    end

    return log(Z)
end

compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Bernoulli) =
    compute_logscale(new_dist, right_dist, left_dist)

function pack_naturalparameters(distribution::Bernoulli)
    p = succprob(distribution)
    return [logit(p)]
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Bernoulli})
    vectorized = getnaturalparameters(ef)
    @inbounds η = vectorized[1]
    return (η, )
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Bernoulli})
    (η, ) = unpack_naturalparameters(exponentialfamily)
    return -log(logistic(-η))
end

function logpartition(::Type{<:Bernoulli}, η)
    return -log(logistic(-first(η)))
end

isproper(exponentialfamily::ExponentialFamilyDistribution{Bernoulli}) = true

check_valid_natural(::Type{<:Bernoulli}, params) = (length(params) === 1)

function support(::Type{<:Bernoulli})
    return SA[0, 1]
end

function support(::ExponentialFamilyDistribution{<:Bernoulli})
    return SA[0, 1]
end

basemeasure(::Type{<:Bernoulli}) = one(Float64)
basemeasure(::ExponentialFamilyDistribution{Bernoulli}) = one(Float64)
basemeasure(::ExponentialFamilyDistribution{Bernoulli}, x) = one(x)
    
sufficientstatistics(type::Type{<:Bernoulli}) = x -> sufficientstatistics(type,x)
sufficientstatistics(::Type{<:Bernoulli}, x::Real) = SA[x]
sufficientstatistics(ef::ExponentialFamilyDistribution{Bernoulli}) = x -> sufficientstatistics(ef,x)
sufficientstatistics(::ExponentialFamilyDistribution{Bernoulli}, x::Real) = SA[x]

function fisherinformation(ef::ExponentialFamilyDistribution{Bernoulli})
    (η, ) = unpack_naturalparameters(ef)
    f = logistic(-η)
    return SA[f * (one(f) - f);;]
end

function fisherinformation(dist::Bernoulli)
    p = dist.p
    return SA[inv(p * (one(p) - p));;]
end

