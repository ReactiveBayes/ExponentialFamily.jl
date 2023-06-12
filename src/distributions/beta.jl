export Beta

import Distributions: Beta, params
import SpecialFunctions: digamma, logbeta, loggamma
import StatsFuns: betalogpdf

vague(::Type{<:Beta}) = Beta(one(Float64), one(Float64))

prod_closed_rule(::Type{<:Beta}, ::Type{<:Beta}) = ClosedProd()

function Base.prod(::ClosedProd, left::Beta, right::Beta)
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

function isproper(params::KnownExponentialFamilyDistribution{Beta})
    αm1 = first(getnaturalparameters(params))
    βm1 = getindex(getnaturalparameters(params), 2)
    return ((αm1 + oneunit(αm1)) > zero(αm1)) && ((βm1 + oneunit(βm1)) > zero(βm1))
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Beta)
    a, b = params(dist)
    KnownExponentialFamilyDistribution(Beta, [a - oneunit(a), b - oneunit(b)])
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Beta})
    params = getnaturalparameters(exponentialfamily)
    αm1    = first(params)
    βm1    = getindex(params, 2)
    return Beta(αm1 + oneunit(αm1), βm1 + oneunit(βm1), check_args = false)
end

check_valid_natural(::Type{<:Beta}, v) = length(v) === 2

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Beta}) =
    logbeta(
        first(getnaturalparameters(exponentialfamily)) + one(Float64),
        getindex(getnaturalparameters(exponentialfamily), 2) + one(Float64)
    )

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Beta}, <:Beta}, x) = 1.0
sufficientstatistics(::Union{<:KnownExponentialFamilyDistribution{Beta}, <:Beta}, x) = [log(x), log(1-x)]
