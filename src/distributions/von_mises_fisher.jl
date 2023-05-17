export VonMisesFisher

import Distributions: VonMisesFisher, params
import SpecialFunctions: besselj

vague(::Type{<:VonMisesFisher}, dims::Int64) = VonMisesFisher(zeros(dims), tiny)

prod_closed_rule(::Type{<:VonMisesFisher}, ::Type{<:VonMisesFisher}) = ClosedProd()

function Distributions.mean(dist::VonMisesFisher)
    (μ, κ) = params(dist)

    p = length(μ)
    factor = besselj(0.5p, κ) / besselj(0.5p - 1, κ)
    return factor * μ
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{VonMisesFisher}) =
    all(0 .<= (getnaturalparameters(exponentialfamily)))

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::VonMisesFisher)
    μ, κ = params(dist)
    KnownExponentialFamilyDistribution(VonMisesFisher, μ * κ)
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{VonMisesFisher})
    κμ = getnaturalparameters(exponentialfamily)
    κ = sqrt(κμ' * κμ)
    μ = κμ / κ
    return VonMisesFisher(μ, κ)
end

check_valid_natural(::Type{<:VonMisesFisher}, v) = length(v) >= 2

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{VonMisesFisher})
    η = getnaturalparameters(exponentialfamily)
    ## because ||μ|| = 1 this trick obtains κ 
    κ = norm(η)
    p = length(η)
    return log(besselj(0.5p - 1, κ))
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{VonMisesFisher}, <:VonMisesFisher}, x) =
    (1 / 2pi)^(length(x) / 2)

function fisherinformation(dist::VonMises)
    _, k = params(dist)
    bessel0 = besseli(0, k)
    bessel1 = besseli(1, k)
    bessel2 = (1 / 2) * (besseli(0, k) + besseli(2, k))
    return [(k)*bessel1/bessel0 0.0; 0.0 bessel2/bessel0-(bessel1/bessel0)^2]
end

function fisherinformation(ef::KnownExponentialFamilyDistribution{VonMisesFisher})
    η = getnaturalparameters(ef)
    η1 = getindex(η, 1)
    η2 = getindex(η, 2)
    u = norm(η)
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
