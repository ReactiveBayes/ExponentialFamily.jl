export Beta

import Distributions: Beta, params
import SpecialFunctions: digamma, logbeta, loggamma, trigamma
import StatsFuns: betalogpdf
using StaticArrays
using LogExpFunctions

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

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Beta})
    vectorized = getnaturalparameters(ef)
    @inbounds η1 = vectorized[1] 
    @inbounds η2 = vectorized[2]
    return η1, η2
end

function isproper(ef::ExponentialFamilyDistribution{Beta})
    αm1,βm1 = unpack_naturalparameters(ef)
    return ((αm1 + one(αm1)) > zero(αm1)) && ((βm1 + one(βm1)) > zero(βm1))
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Beta)
    ExponentialFamilyDistribution(Beta, pack_naturalparameters(dist))
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Beta})
    αm1 , βm1 = unpack_naturalparameters(exponentialfamily)
    return Beta(αm1 + one(αm1), βm1 + one(βm1), check_args = false)
end

check_valid_natural(::Type{<:Beta}, v) = length(v) === 2

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Beta}) 
    αm1 , βm1 = unpack_naturalparameters(exponentialfamily)
    return logbeta(
        αm1 + one(αm1),
        βm1 + one(βm1)
    )
end
function support(::ExponentialFamilyDistribution{Beta})
    return ClosedInterval{Real}(zero(Float64), one(Float64))
end

function basemeasure(::ExponentialFamilyDistribution{Beta}, x::Real)
    @assert Distributions.insupport(Beta, x) "basemeasure for Beta should be evaluated at positive values"
    return one(x)
end
function sufficientstatistics(ef::ExponentialFamilyDistribution{Beta}, x::Real)
    @assert insupport(ef, x) "sufficientstatistics for Beta should be evaluated at positive values"
    return  SA[log(x), log(one(x) - x)]
end

function fisherinformation(dist::Beta)
    a, b = params(dist)
    psia = trigamma(a)
    psib = trigamma(b)
    psiab = trigamma(a + b)

    return SA[psia-psiab -psiab; -psiab psib-psiab]
end

function fisherinformation(ef::ExponentialFamilyDistribution{Beta})
    η1, η2 = unpack_naturalparameters(ef)

    psia = trigamma(η1 + one(typeof(η1)))
    psib = trigamma(η2 + one(typeof(η2)))
    psiab = trigamma(η1 + η2 + 2)
    return SA[psia-psiab -psiab; -psiab psib-psiab]
end
