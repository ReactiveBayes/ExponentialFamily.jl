export Beta

import Distributions: Beta, params
import SpecialFunctions: digamma, logbeta, loggamma, trigamma
import StatsFuns: betalogpdf
using StaticArrays

vague(::Type{<:Beta}) = Beta(one(Float64), one(Float64))

closed_prod_rule(::Type{<:Beta}, ::Type{<:Beta}) = ClosedProd()

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

function pack_naturalparameters(dist::Beta)
    a, b = params(dist)
    return [a - oneunit(a), b - oneunit(b)]
end

function unpack_naturalparameters(ef::KnownExponentialFamilyDistribution{<:Beta})
    vectorized = getnaturalparameters(ef)
    @inbounds η1 = vectorized[1] 
    @inbounds η2 = vectorized[2]
    return η1, η2
end

function isproper(ef::KnownExponentialFamilyDistribution{Beta})
    αm1,βm1 = unpack_naturalparameters(ef)
    return ((αm1 + oneunit(αm1)) > zero(αm1)) && ((βm1 + oneunit(βm1)) > zero(βm1))
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Beta)
    KnownExponentialFamilyDistribution(Beta, pack_naturalparameters(dist))
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Beta})
    αm1 , βm1 = unpack_naturalparameters(exponentialfamily)
    return Beta(αm1 + oneunit(αm1), βm1 + oneunit(βm1), check_args = false)
end

check_valid_natural(::Type{<:Beta}, v) = length(v) === 2

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Beta}) 
    αm1 , βm1 = unpack_naturalparameters(exponentialfamily)
    return logbeta(
        αm1 + one(Float64),
        βm1 + one(Float64)
    )
end
function support(::KnownExponentialFamilyDistribution{Beta})
    return ClosedInterval{Real}(zero(Float64), one(Float64))
end

function basemeasure(::KnownExponentialFamilyDistribution{Beta}, x)
    @assert Distributions.insupport(Beta, x) "basemeasure for Beta should be evaluated at positive values"
    return one(typeof(x))
end
function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Beta}, x)
    @assert insupport(ef, x) "sufficientstatistics for Beta should be evaluated at positive values"
    return  SA[log(x), log(one(x) - x)]
end

function fisherinformation(dist::Beta)
    a, b = params(dist)
    psia = trigamma(a)
    psib = trigamma(b)
    psiab = trigamma(a + b)

    return [psia-psiab -psiab; -psiab psib-psiab]
end

function fisherinformation(ef::KnownExponentialFamilyDistribution{Beta})
    η1, η2 = unpack_naturalparameters(ef)

    psia = trigamma(η1 + one(typeof(η1)))
    psib = trigamma(η2 + one(typeof(η2)))
    psiab = trigamma(η1 + η2 + 2)
    return [psia-psiab -psiab; -psiab psib-psiab]
end
