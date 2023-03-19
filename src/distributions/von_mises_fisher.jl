export VonMisesFisher

import Distributions: VonMisesFisher, params
import SpecialFunctions: besselj

vague(::Type{<:VonMisesFisher}, dims::Int64) = VonMisesFisher(zeros(dims), tiny)

prod_analytical_rule(::Type{<:VonMisesFisher}, ::Type{<:VonMisesFisher}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::VonMisesFisher, right::VonMisesFisher)
    ef_left = Base.convert(ExponentialFamilyDistribution, left)
    ef_right = Base.convert(ExponentialFamilyDistribution, right)
    naturalparams = getnaturalparameters(ef_left) + getnaturalparameters(ef_right)
    return Base.convert(Distribution, ExponentialFamilyDistribution(VonMisesFisher,naturalparams))
end

function Distributions.mean(dist::VonMisesFisher)
    (μ, κ) = params(dist)

    p = length(μ)
    factor = besselj(0.5p, κ) / besselj(0.5p - 1, κ)
    return factor * μ
end

isproper(exponentialfamily::ExponentialFamilyDistribution{VonMisesFisher}) = all(0 .<= (getnaturalparameters(exponentialfamily)))

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::VonMisesFisher)
    μ, κ = params(dist)
    ExponentialFamilyDistribution(VonMisesFisher, μ * κ)
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{VonMisesFisher})
    κμ = getnaturalparameters(exponentialfamily)
    κ = sqrt(κμ' * κμ)
    μ = κμ / κ
    return VonMisesFisher(μ, κ)
end

check_valid_natural(::Type{<:VonMisesFisher}, v) = length(v) >= 2

function lognormalizer(exponentialfamily::ExponentialFamilyDistribution{VonMisesFisher})
    η = getnaturalparameters(exponentialfamily)
    ## because cos^2+sin^2 = 1 this trick obtains κ 
    κ = sqrt(η' * η)
    p = length(η)
    return log(besselj(0.5p - 1, κ))
end
basemeasure(::Union{<:ExponentialFamilyDistribution{VonMisesFisher}, <:VonMisesFisher}, x) = (1 / 2pi)^(length(x) / 2)

plus(::ExponentialFamilyDistribution{VonMisesFisher}, ::ExponentialFamilyDistribution{VonMisesFisher}) = Plus()
