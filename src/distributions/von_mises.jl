export VonMises

import Distributions: VonMises, params
import SpecialFunctions: besselj0

vague(::Type{<:VonMises}) = VonMises(0.0, tiny)

prod_analytical_rule(::Type{<:VonMises}, ::Type{<:VonMises}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::VonMises, right::VonMises)
    naturalparams_left = Base.convert(ExponentialFamilyDistribution, left)
    naturalparams_right = Base.convert(ExponentialFamilyDistribution, right)
    naturalparams = naturalparams_left + naturalparams_right
    return Base.convert(Distribution, naturalparams)
end

isproper(params::ExponentialFamilyDistribution{VonMises}) = true

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::VonMises)
    μ, κ = params(dist)
    ExponentialFamilyDistribution(VonMises, [κ * cos(μ), κ * sin(μ)])
end

function Base.convert(::Type{Distribution}, η::ExponentialFamilyDistribution{VonMises})
    params = getnaturalparameters(η)
    κcosμ  = first(params)

    κ = sqrt(params' * params)
    μ = acos(κcosμ / κ)
    return VonMises(μ, κ)
end

check_valid_natural(::Type{<:VonMises}, v) = length(v) === 2

function lognormalizer(params::ExponentialFamilyDistribution{VonMises})
    η = getnaturalparameters(params)
    κ = sqrt(η' * η)
    return log(besselj0(κ))
end
basemeasure(::Union{<:ExponentialFamilyDistribution{VonMises}, <:VonMises}, x) = 1 / 2pi
plus(::ExponentialFamilyDistribution{VonMises}, ::ExponentialFamilyDistribution{VonMises}) = Plus()
