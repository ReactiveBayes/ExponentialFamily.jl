export Erlang

import SpecialFunctions: logfactorial, digamma
import Distributions: Erlang, shape, scale, cov
using StaticArrays

BayesBase.cov(dist::Erlang) = var(dist)

function BayesBase.mean(::typeof(log), dist::Erlang)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

BayesBase.vague(::Type{<:Erlang}) = Erlang(1, huge)
BayesBase.default_prod_rule(::Type{<:Erlang}, ::Type{<:Erlang}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::Erlang, right::Erlang)
    return Erlang(shape(left) + shape(right) - 1, (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

# Natural parameterization

isproper(::NaturalParametersSpace, ::Type{Erlang}, η, conditioner) =
    isnothing(conditioner) && length(η) === 2 && isinteger(η[1]) && (η[1] >= 0) && (-η[2] >= tiny) && all(!isinf, η)
isproper(::MeanParametersSpace, ::Type{Erlang}, θ, conditioner) = isnothing(conditioner) && length(θ) === 2 && all(>(0), θ) && isinteger(θ[1]) && all(!isinf, θ)

function (::MeanToNatural{Erlang})(tuple_of_θ::Tuple{Any, Any})
    (shape, scale) = tuple_of_θ
    return (shape - one(shape), -inv(scale))
end

function (::NaturalToMean{Erlang})(tuple_of_η::Tuple{Any, Any})
    (η1, η2) = tuple_of_η
    return (η1 + one(η1), -inv(η2))
end

function unpack_parameters(::Type{Erlang}, packed)
    fi = firstindex(packed)
    si = fi + 1
    return (packed[fi], packed[si])
end

function convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{Erlang})
    tuple_of_η = unpack_parameters(ef)
    params = map(NaturalParametersSpace() => MeanParametersSpace(), Erlang, tuple_of_η)
    return Erlang(Integer(params[1]), params[2])
end

isbasemeasureconstant(::Type{Erlang}) = ConstantBaseMeasure()

getbasemeasure(::Type{Erlang}) = (x) -> one(x)
getsufficientstatistics(::Type{Erlang}) = (log, identity)

getlogpartition(::NaturalParametersSpace, ::Type{Erlang}) = (η) -> begin
    (η1, η2) = unpack_parameters(Erlang, η)

    return loggamma(η1 + 1) - (η1 + one(η1)) * log(-η2)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Erlang}) = (η) -> begin
    (η1, η2) = unpack_parameters(Erlang, η)
    dη1 = digamma(η1 + 1) - log(-η2)
    dη2 = -(η1 + one(η1)) * inv(η2)
    return SA[dη1, dη2]
end

getfisherinformation(::NaturalParametersSpace, ::Type{Erlang}) = (η) -> begin
    (η1, η2) = unpack_parameters(Erlang, η)
    miη2 = -inv(η2)

    return SA[trigamma(η1 + 1) miη2; miη2 (η1+1)/(η2^2)]
end

# Mean parameterization

getlogpartition(::MeanParametersSpace, ::Type{Erlang}) = (θ) -> begin
    (k, β) = unpack_parameters(Erlang, θ)
    return k * log(β) + logfactorial(k - 1)
end

getfisherinformation(::MeanParametersSpace, ::Type{Erlang}) = (θ) -> begin
    (k, β) = unpack_parameters(Erlang, θ)
    return SA[trigamma(k) 1/β; 1/β k/β^2]
end
