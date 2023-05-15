export VonMises

import Distributions: VonMises, params
import SpecialFunctions: besselj0

vague(::Type{<:VonMises}) = VonMises(0.0, tiny)

prod_closed_rule(::Type{<:VonMises}, ::Type{<:VonMises}) = ClosedProd()

isproper(params::KnownExponentialFamilyDistribution{VonMises}) = true

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::VonMises)
    μ, κ = params(dist)
    KnownExponentialFamilyDistribution(VonMises, [κ * cos(μ), κ * sin(μ)])
end

function Base.convert(::Type{Distribution}, ef::KnownExponentialFamilyDistribution{VonMises})
    params = getnaturalparameters(ef)
    κcosμ  = first(params)

    κ = sqrt(params' * params)
    μ = acos(κcosμ / κ)
    return VonMises(μ, κ)
end

check_valid_natural(::Type{<:VonMises}, v) = length(v) === 2

function logpartition(params::KnownExponentialFamilyDistribution{VonMises})
    η = getnaturalparameters(params)
    κ = sqrt(η' * η)
    return log(besseli(0,κ))
end
basemeasure(::Union{<:KnownExponentialFamilyDistribution{VonMises}, <:VonMises}, x) = 1 / 2pi

function fisherinformation(dist::VonMises)
    _, k = params(dist)
    return [0 0; 0 besseli(2,k)/besseli(0,k) - (besseli(1,k)/besseli(0,k))^2]
end

function fisherinformation(ef::KnownExponentialFamilyDistribution{VonMises})
    params = getnaturalparameters(ef)
    k = sqrt(params' * params)
    return besseli(2,k)/besseli(0,k) - (besseli(1,k)/besseli(0,k))^2
end
