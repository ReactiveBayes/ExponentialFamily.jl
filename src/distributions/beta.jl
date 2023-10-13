export Beta

import Distributions: Beta, params
import SpecialFunctions: digamma, logbeta, loggamma, trigamma
import StatsFuns: betalogpdf

using StaticArrays
using LogExpFunctions

BayesBase.vague(::Type{<:Beta}) = Beta(one(Float64), one(Float64))
BayesBase.default_prod_rule(::Type{<:Beta}, ::Type{<:Beta}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::Beta, right::Beta)
    left_a, left_b   = params(left)
    right_a, right_b = params(right)
    T                = promote_paramfloattype(left, right)
    return Beta(left_a + right_a - one(T), left_b + right_b - one(T))
end

function BayesBase.compute_logscale(new_dist::Beta, left_dist::Beta, right_dist::Beta)
    return logbeta(params(new_dist)...) - logbeta(params(left_dist)...) - logbeta(params(right_dist)...)
end

function BayesBase.mean(::typeof(log), dist::Beta)
    a, b = params(dist)
    return digamma(a) - digamma(a + b)
end

function BayesBase.mean(::typeof(mirrorlog), dist::Beta)
    a, b = params(dist)
    return digamma(b) - digamma(a + b)
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Beta}, η, conditioner) = isnothing(conditioner) && length(η) === 2 && all(>(-1), η)
isproper(::MeanParametersSpace, ::Type{Beta}, θ, conditioner) = isnothing(conditioner) && length(θ) === 2 && all(>(0), θ)

function (::MeanToNatural{Beta})(tuple_of_θ::Tuple{Any, Any})
    (a, b) = tuple_of_θ
    return (a - one(a), b - one(b))
end

function (::NaturalToMean{Beta})(tuple_of_η::Tuple{Any, Any})
    (η₁, η₂) = tuple_of_η
    return (η₁ + one(η₁), η₂ + one(η₂))
end

function unpack_parameters(::Type{Beta}, packed)
    fi = firstindex(packed)
    si = firstindex(packed) + 1
    return (packed[fi], packed[si])
end

isbasemeasureconstant(::Type{Beta}) = ConstantBaseMeasure()

getbasemeasure(::Type{Beta}) = (x) -> oneunit(x)
getsufficientstatistics(::Type{Beta}) = (log, mirrorlog)

getlogpartition(::NaturalParametersSpace, ::Type{Beta}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(Beta, η)
    return logbeta(η₁ + one(η₁), η₂ + one(η₂))
end

getfisherinformation(::NaturalParametersSpace, ::Type{Beta}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(Beta, η)
    psia = trigamma(η₁ + one(η₁))
    psib = trigamma(η₂ + one(η₂))
    psiab = trigamma(η₁ + η₂ + 2)
    return SA[psia-psiab -psiab; -psiab psib-psiab]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Beta}) = (θ) -> begin
    (a, b) = unpack_parameters(Beta, θ)
    return logbeta(a, b)
end

getfisherinformation(::MeanParametersSpace, ::Type{Beta}) = (θ) -> begin
    (a, b) = unpack_parameters(Beta, θ)
    psia = trigamma(a)
    psib = trigamma(b)
    psiab = trigamma(a + b)

    return SA[psia-psiab -psiab; -psiab psib-psiab]
end
