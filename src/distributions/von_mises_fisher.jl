export VonMisesFisher

import Distributions: VonMisesFisher, params
import SpecialFunctions: besseli

vague(::Type{<:VonMisesFisher}, dims::Int64) = VonMisesFisher(zeros(dims), tiny)

prod_closed_rule(::Type{<:VonMisesFisher}, ::Type{<:VonMisesFisher}) = ClosedProd()

function Distributions.mean(dist::VonMisesFisher)
    (μ, κ) = params(dist)

    p = length(μ)
    factor = besseli(0.5p, κ) / besseli(0.5p - 1, κ)
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
    κ = sqrt(η' * η)
    p = length(η)
    return log(besseli((p / 2) - 1, κ)) - ((p / 2) - 1) * log(κ)
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{VonMisesFisher}, <:VonMisesFisher}, x) =
    (1 / 2pi)^(length(x) / 2)

function fisherinformation(dist::VonMisesFisher)
    μ, k = params(dist)
    p    = length(μ)

    bessel3 = besseli(p / 2 - 3, k)
    bessel2 = besseli(p / 2 - 2, k)
    bessel1 = besseli(p / 2 - 1, k)
    bessel0 = besseli(p / 2, k)
    bessel4 = besseli(p / 2 + 1, k)

    tmp =
        (p / 2 - 1) / k^2 + (1 / 4) * (bessel3 + 2 * bessel1 + bessel4) / bessel1 -
        (1 / 4) * (bessel2 + bessel0)^2 / bessel1^2
    Ap = bessel0 / bessel1
    tmp2 = (1 - Ap * p / k - Ap^2) * μ * μ' + inv(k) * Ap * diageye(p)
    return [k^2*tmp2 -Ap*μ; -Ap*μ' tmp]
end

function fisherinformation(ef::KnownExponentialFamilyDistribution{VonMisesFisher})
    η = getnaturalparameters(ef)
    u = norm(η)
    p = length(η)

    bessel3 = besseli(p / 2 - 3, u)
    bessel2 = besseli(p / 2 - 2, u)
    bessel1 = besseli(p / 2 - 1, u)
    bessel0 = besseli(p / 2, u)
    bessel4 = besseli(p / 2 + 1, u)

    f1 = (1 / 2) * (bessel0 + bessel2)
    f2 = inv(bessel1)
    f3 = (p / 2 - 1) / u
    f4 = η / u

    delu = η' / u
    df1  = (1 / 4) * (bessel4 + 2 * bessel1 + bessel3) * delu
    df2  = ((-1 / 2) * (bessel2 + bessel0) / bessel1^2) * delu
    df3  = (-(p / 2 - 1) / u^2) * delu
    df4  = diageye(p) / u - η * η' / u^3

    return f4 * df1 * f2 + f4 * f1 * df2 + f1 * f2 * df4 - f4 * df3 - f3 * df4
end
