export VonMises
using Distributions
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
    return log(besseli(0, κ))
end
basemeasure(::Union{<:KnownExponentialFamilyDistribution{VonMises}, <:VonMises}, x) = 1 / 2pi

function fisherinformation(dist::VonMises)
    _, k = params(dist)
    bessel0 = besseli(0, k)
    bessel1 = besseli(1, k)
    bessel2 = (1 / 2) * (besseli(0, k) + besseli(2, k))
    return [(k)*bessel1/bessel0 0.0; 0.0 bessel2/bessel0-(bessel1/bessel0)^2]
end

function fisherinformation(ef::KnownExponentialFamilyDistribution{VonMises})
    η = getnaturalparameters(ef)
    η1 = getindex(η, 1)
    η2 = getindex(η, 2)
    u = sqrt(η' * η)
    bessel0 = besseli(0, u)
    bessel1 = besseli(1, u)
    bessel2 = (1 / 2) * (besseli(0, u) + besseli(2, u))

    h11 =
        (bessel2 / bessel0) * (η1^2 / u^2) - (bessel1 / bessel0)^2 * (η1^2 / u^2) +
        (bessel1 / bessel0) * (1 / u - (η1^2 / u^3))
    h22 =
        (bessel2 / bessel0) * (η2^2 / u^2) - (bessel1 / bessel0)^2 * (η2^2 / u^2) +
        (bessel1 / bessel0) * (1 / u - (η2^2 / u^3))
    h12 = (η1 * η2 / u^2) * (bessel2 / bessel0 - (bessel1 / bessel0)^2 - bessel1 / (u * bessel0))

    return [h11 h12; h12 h22]
end

sufficientstatistics(::Union{<:KnownExponentialFamilyDistribution{VonMises}, <:VonMises}, x) = [cos(x), sin(x)]