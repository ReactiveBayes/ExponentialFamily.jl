export VonMisesFisher

import Distributions: VonMisesFisher, params
import SpecialFunctions: besselj

vague(::Type{<:VonMisesFisher}, dims::Int64) = VonMisesFisher(zeros(dims), tiny)

prod_analytical_rule(::Type{<:VonMisesFisher}, ::Type{<:VonMisesFisher}) = ClosedProd()

# function Base.prod(::ClosedProd, left::VonMisesFisher, right::VonMisesFisher)
#     ef_left = Base.convert(KnownExponentialFamilyDistribution, left)
#     ef_right = Base.convert(KnownExponentialFamilyDistribution, right)
#     naturalparams = getnaturalparameters(ef_left) + getnaturalparameters(ef_right)
#     return Base.convert(Distribution, KnownExponentialFamilyDistribution(VonMisesFisher,naturalparams))
# end

function Distributions.mean(dist::VonMisesFisher)
    (μ, κ) = params(dist)

    p = length(μ)
    factor = besselj(0.5p, κ) / besselj(0.5p - 1, κ)
    return factor * μ
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{VonMisesFisher}) = all(0 .<= (getnaturalparameters(exponentialfamily)))

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
    ## because cos^2+sin^2 = 1 this trick obtains κ 
    κ = sqrt(η' * η)
    p = length(η)
    return log(besselj(0.5p - 1, κ))
end
basemeasure(::Union{<:KnownExponentialFamilyDistribution{VonMisesFisher}, <:VonMisesFisher}, x) = (1 / 2pi)^(length(x) / 2)

