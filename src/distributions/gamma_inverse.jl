export GammaInverse

import Distributions: InverseGamma, shape, scale
import SpecialFunctions: digamma, polygamma
import ForwardDiff

const GammaInverse = InverseGamma

BayesBase.vague(::Type{<:GammaInverse}) = InverseGamma(2.0, huge)
BayesBase.default_prod_rule(::Type{<:GammaInverse}, ::Type{<:GammaInverse}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::GammaInverse, right::InverseGamma)
    return GammaInverse(shape(left) + shape(right) + one(Float64), scale(left) + scale(right))
end

function BayesBase.mean(::typeof(log), dist::GammaInverse)
    α = shape(dist)
    θ = scale(dist)
    return log(θ) - digamma(α)
end

function BayesBase.mean(::typeof(inv), dist::GammaInverse)
    α = shape(dist)
    θ = scale(dist)
    return α / θ
end

# Natural parametrization

isproper(::MeanParametersSpace, ::Type{GammaInverse}, θ, conditioner) = isnothing(conditioner) && length(θ) === 2 && all(>(0), θ)

function isproper(::NaturalParametersSpace, ::Type{GammaInverse}, η, conditioner)
    if length(η) !== 2
        return false
    end
    (η₁, η₂) = unpack_parameters(GammaInverse, η)
    return isnothing(conditioner) && (η₁ < -1) && (η₂ < 0)
end

function (::MeanToNatural{GammaInverse})(tuple_of_θ::Tuple{Any, Any})
    (shape, scale) = tuple_of_θ
    return (-shape - 1, -scale)
end

function (::NaturalToMean{GammaInverse})(tuple_of_η::Tuple{Any, Any})
    (η₁, η₂) = tuple_of_η
    return (-η₁ - 1, -η₂)
end

function unpack_parameters(::Type{GammaInverse}, packed)
    fi = firstindex(packed)
    si = firstindex(packed) + 1
    return (packed[fi], packed[si])
end

isbasemeasureconstant(::Type{GammaInverse}) = ConstantBaseMeasure()

getbasemeasure(::Type{GammaInverse}) = (x) -> oneunit(x)
getsufficientstatistics(::Type{GammaInverse}) = (log, inv)

getexpectationlogbasemeasure(::NaturalParametersSpace, ::Type{GammaInverse}) = (_) -> begin
    log(1)
end

getlogpartition(::NaturalParametersSpace, ::Type{GammaInverse}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(GammaInverse, η)
    return loggamma(-η₁ - one(η₁)) - (-η₁ - one(η₁)) * log(-η₂)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{GammaInverse}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(GammaInverse, η)
    dη1 = -digamma(-η₁ - one(η₁)) + log(-η₂)
    dη2 = - (-η₁ - one(η₁))/η₂
    return SA[dη1, dη2]
end

getfisherinformation(::NaturalParametersSpace, ::Type{GammaInverse}) =
    (η) -> begin
        (η₁, η₂) = unpack_parameters(GammaInverse, η)
        return SA[polygamma(1, -η₁ - one(η₁)) -inv(-η₂); -inv(-η₂) (-η₁-one(η₁))/(η₂^2)]
    end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{GammaInverse}) = (θ) -> begin
    (shape, scale) = unpack_parameters(Gamma, θ)
    return loggamma(shape) - shape * log(scale)
end

getfisherinformation(::MeanParametersSpace, ::Type{GammaInverse}) =
    (θ) -> begin
        (shape, scale) = unpack_parameters(Gamma, θ)
        return SA[polygamma(1, shape) -inv(scale); -inv(scale) shape/(scale^2)]
    end
