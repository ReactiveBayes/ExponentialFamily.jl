export VonMises
using Distributions
import Distributions: VonMises, params
import SpecialFunctions: besselj0

vague(::Type{<:VonMises}) = VonMises(0.0, tiny)

closed_prod_rule(::Type{<:VonMises}, ::Type{<:VonMises}) = ClosedProd()

isproper(params::ExponentialFamilyDistribution{VonMises}) = true

function pack_naturalparameters(dist::VonMises)
    μ, κ = params(dist)
    return [κ * cos(μ), κ * sin(μ)]
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:VonMises})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    @inbounds η2 = η[2]

    return η1, η2
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::VonMises) = ExponentialFamilyDistribution(VonMises, pack_naturalparameters(dist))
    
function Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{VonMises})
    η1, η2 = unpack_naturalparameters(ef)
    κ = sqrt(η1^2 + η2^2)
    μ = acos(η1 / κ)
    return VonMises(μ, κ)
end

check_valid_natural(::Type{<:VonMises}, v) = length(v) === 2

function logpartition(params::ExponentialFamilyDistribution{VonMises})
    η = getnaturalparameters(params)
    κ = sqrt(η' * η)
    return log(besseli(0, κ))
end

function fisherinformation(dist::VonMises)
    _, k = params(dist)
    bessel0 = besseli(0, k)
    bessel1 = besseli(1, k)
    bessel2 = (1 / 2) * (besseli(0, k) + besseli(2, k))
    return SA[(k)*bessel1/bessel0 0.0; 0.0 bessel2/bessel0-(bessel1/bessel0)^2]
end

function fisherinformation(ef::ExponentialFamilyDistribution{VonMises})
    η1,η2 = unpack_naturalparameters(ef)
    u = sqrt(η1^2 + η2^2)
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

    return SA[h11 h12; h12 h22]
end

sufficientstatistics(ef::ExponentialFamilyDistribution{VonMises}) = (x) -> sufficientstatistics(ef,x)
function sufficientstatistics(::Union{<:ExponentialFamilyDistribution{VonMises}, <:VonMises}, x::Real)
    return SA[cos(x), sin(x)]
end

basemeasure(::ExponentialFamilyDistribution{VonMises}) = inv(TWOPI)
basemeasure(::Union{<:ExponentialFamilyDistribution{VonMises}, <:VonMises}, x::Real) = inv(TWOPI)

