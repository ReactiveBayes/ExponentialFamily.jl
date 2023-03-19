export Beta

import Distributions: Beta, params
import SpecialFunctions: digamma, logbeta, loggamma
import StatsFuns: betalogpdf

vague(::Type{<:Beta}) = Beta(1.0, 1.0)

prod_analytical_rule(::Type{<:Beta}, ::Type{<:Beta}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Beta, right::Beta)
    left_a, left_b   = params(left)
    right_a, right_b = params(right)
    T                = promote_samplefloattype(left, right)
    return Beta(left_a + right_a - one(T), left_b + right_b - one(T))
end

function compute_logscale(new_dist::Beta, left_dist::Beta, right_dist::Beta)
    return logbeta(params(new_dist)...) - logbeta(params(left_dist)...) - logbeta(params(right_dist)...)
end

function mean(::typeof(log), dist::Beta)
    a, b = params(dist)
    return digamma(a) - digamma(a + b)
end

function mean(::typeof(mirrorlog), dist::Beta)
    a, b = params(dist)
    return digamma(b) - digamma(a + b)
end

function isproper(params::ExponentialFamilyDistribution{Beta})
    αm1 = first(getnaturalparameters(params))
    βm1 = getindex(getnaturalparameters(params), 2)
    return ((αm1 + 1) > 0) && ((βm1 + 1) > 0)
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Beta)
    a, b = params(dist)
    ExponentialFamilyDistribution(Beta, [a - 1, b - 1])
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Beta})
    params = getnaturalparameters(exponentialfamily)
    αm1    = first(params)
    βm1    = getindex(params, 2)
    return Beta(αm1 + 1, βm1 + 1, check_args = false)
end

check_valid_natural(::Type{<:Beta}, v) = length(v) === 2

lognormalizer(exponentialfamily::ExponentialFamilyDistribution{Beta}) =
    logbeta(first(getnaturalparameters(exponentialfamily)) + 1, getindex(getnaturalparameters(exponentialfamily), 2) + 1)

basemeasure(T::Union{<:ExponentialFamilyDistribution{Beta}, <:Beta}, x) = 1.0

plus(::ExponentialFamilyDistribution{Beta}, ::ExponentialFamilyDistribution{Beta}) = Plus()
