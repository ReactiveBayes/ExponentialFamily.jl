export VonMises

import Distributions: VonMises, params
import SpecialFunctions: besselj0

vague(::Type{<:VonMises}) = VonMises(0.0, tiny)

prod_analytical_rule(::Type{<:VonMises}, ::Type{<:VonMises}) = ClosedProd()

isproper(params::KnownExponentialFamilyDistribution{VonMises}) = true

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::VonMises)
    μ, κ = params(dist)
    KnownExponentialFamilyDistribution(VonMises, [κ * cos(μ), κ * sin(μ)])
end

function Base.convert(::Type{Distribution}, η::KnownExponentialFamilyDistribution{VonMises})
    params = getnaturalparameters(η)
    κcosμ  = first(params)

    κ = sqrt(params' * params)
    μ = acos(κcosμ / κ)
    return VonMises(μ, κ)
end

check_valid_natural(::Type{<:VonMises}, v) = length(v) === 2

function logpartition(params::KnownExponentialFamilyDistribution{VonMises})
    η = getnaturalparameters(params)
    κ = sqrt(η' * η)
    return log(besselj0(κ))
end
basemeasure(::Union{<:KnownExponentialFamilyDistribution{VonMises}, <:VonMises}, x) = 1 / 2pi
