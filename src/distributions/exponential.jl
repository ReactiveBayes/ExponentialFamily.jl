export Exponential

import Distributions: Exponential, params
import SpecialFunctions: digamma, logbeta
using StaticArrays

BayesBase.vague(::Type{<:Exponential}) = Exponential(Float64(huge))
BayesBase.default_prod_rule(::Type{<:Exponential}, ::Type{<:Exponential}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::Exponential, right::Exponential)
    invθ_left  = inv(left.θ)
    invθ_right = inv(right.θ)
    return Exponential(inv(invθ_left + invθ_right))
end

function BayesBase.mean(::typeof(log), dist::Exponential)
    return -log(rate(dist)) - MathConstants.eulergamma
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Exponential}, η, conditioner) = isnothing(conditioner) && length(η) === 1 && first(η) < 0
isproper(::MeanParametersSpace, ::Type{Exponential}, θ, conditioner) = isnothing(conditioner) && length(θ) === 1 && first(θ) > 0

function (::MeanToNatural{Exponential})(tuple_of_θ::Tuple{Any})
    (scale,) = tuple_of_θ
    return (-inv(scale),)
end

function (::NaturalToMean{Exponential})(tuple_of_η::Tuple{Any})
    (η₁,) = tuple_of_η
    return (-inv(η₁),)
end

unpack_parameters(::Type{Exponential}, packed) = (first(packed),)

isbasemeasureconstant(::Type{Exponential}) = ConstantBaseMeasure()

getbasemeasure(::Type{Exponential}) = (x) -> oneunit(x)
getsufficientstatistics(::Type{Exponential}) = (identity,)

getlogpartition(::NaturalParametersSpace, ::Type{Exponential}) = (η) -> begin
    (η₁,) = unpack_parameters(Exponential, η)
    return -log(-η₁)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Exponential}) = (η) -> begin
    (η₁,) = unpack_parameters(Exponential, η)
    return SA[-1/η₁]
end

getfisherinformation(::NaturalParametersSpace, ::Type{Exponential}) = (η) -> begin
    (η₁,) = unpack_parameters(Exponential, η)
    SA[inv(η₁^2);;]
end

## Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Exponential}) = (θ) -> begin
    (scale,) = unpack_parameters(Exponential, θ)
    return log(scale)
end

getfisherinformation(::MeanParametersSpace, ::Type{Exponential}) = (θ) -> begin
    (scale,) = unpack_parameters(Exponential, θ)
    SA[inv(scale^2);;]
end
