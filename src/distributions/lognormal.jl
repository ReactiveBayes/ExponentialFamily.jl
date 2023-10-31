export LogNormal

import SpecialFunctions: digamma
import Distributions: LogNormal
using StaticArrays

BayesBase.cov(dist::LogNormal) = var(dist)
BayesBase.vague(::Type{<:LogNormal}) = LogNormal(1, 1e12)

BayesBase.default_prod_rule(::Type{<:LogNormal}, ::Type{<:LogNormal}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::LogNormal, right::LogNormal)
    mean1, scale1 = params(left)
    mean2, scale2 = params(right)
    var1 = scale1^2
    var2 = scale2^2

    var3 = (var1 * var2) / (var1 + var2)
    mean3 = var3 * (mean1 / var1 + mean2 / var2 - 1)

    return LogNormal(mean3, sqrt(var3))
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{LogNormal}, η, conditioner) = isnothing(conditioner) && (length(η) === 2) && (η[firstindex(η)+1] < 0)
isproper(::MeanParametersSpace, ::Type{LogNormal}, θ, conditioner) = isnothing(conditioner) && (length(θ) === 2) && (θ[firstindex(θ)+1] > 0)

function (::MeanToNatural{LogNormal})(tuple_of_θ::Tuple{Any, Any})
    (μ, σ) = tuple_of_θ
    σ² = abs2(σ)
    return (μ / σ² - 1, -1 / (2σ²))
end

function (::NaturalToMean{LogNormal})(tuple_of_η::Tuple{Any, Any})
    (η₁, η₂) = tuple_of_η
    return (-(η₁ + 1) / (2η₂), sqrt(-1 / (2η₂)))
end

function unpack_parameters(::Type{LogNormal}, packed)
    fi = firstindex(packed)
    si = firstindex(packed) + 1
    return (packed[fi], packed[si])
end

isbasemeasureconstant(::Type{LogNormal}) = ConstantBaseMeasure()

getbasemeasure(::Type{LogNormal}) = (x) -> invsqrt2π
getsufficientstatistics(::Type{LogNormal}) = (log, x -> abs2(log(x)))

getlogpartition(::NaturalParametersSpace, ::Type{LogNormal}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(LogNormal, η)
    return -(η₁ + 1)^2 / (4η₂) - log(-2η₂) / 2
end

getfisherinformation(::NaturalParametersSpace, ::Type{LogNormal}) =
    (η) -> begin
        (η₁, η₂) = unpack_parameters(LogNormal, η)
        return SA[-1/(2η₂) (η₁+1)/(2η₂^2); (η₁+1)/(2η₂^2) -(η₁ + 1)^2/(2*(η₂^3))+1/(2*η₂^2)]
    end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{LogNormal}) = (θ) -> begin
    (μ, σ) = unpack_parameters(LogNormal, θ)
    return abs2(μ) / (2abs2(σ)) + log(σ)
end

getfisherinformation(::MeanParametersSpace, ::Type{LogNormal}) = (θ) -> begin
    (μ, σ) = unpack_parameters(LogNormal, θ)
    invσ² = inv(abs2(σ))
    return SA[invσ² 0.0; 0.0 2invσ²]
end
