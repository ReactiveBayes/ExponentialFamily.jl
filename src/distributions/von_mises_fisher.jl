export VonMisesFisher

import Distributions: VonMisesFisher, params
import SpecialFunctions: besselj

vague(::Type{<:VonMisesFisher}, dims::Int64) = VonMisesFisher(zeros(dims), tiny)

prod_analytical_rule(::Type{<:VonMisesFisher}, ::Type{<:VonMisesFisher}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::VonMisesFisher, right::VonMisesFisher)
    naturalparams_left = Base.convert(NaturalParameters, left)
    naturalparams_right = Base.convert(NaturalParameters, right)
    naturalparams = naturalparams_left + naturalparams_right
    return Base.convert(Distribution, naturalparams)
end

function Distributions.mean(dist::VonMisesFisher)
    (μ, κ) = params(dist)

    p = length(μ)
    factor = besselj(0.5p, κ) / besselj(0.5p - 1, κ)
    return factor * μ
end

isproper(params::NaturalParameters{VonMisesFisher}) = all(0 .<= (get_params(params)))

function Base.convert(::Type{NaturalParameters}, dist::VonMisesFisher)
    μ, κ = params(dist)
    NaturalParameters(VonMisesFisher, μ * κ)
end

function Base.convert(::Type{Distribution}, η::NaturalParameters{VonMisesFisher})
    κμ = get_params(η)
    κ = sqrt(κμ' * κμ)
    μ = κμ / κ
    return VonMisesFisher(μ, κ)
end

check_valid_natural(::Type{<:VonMisesFisher}, v) = length(v) >= 2

function lognormalizer(params::NaturalParameters{VonMisesFisher})
    η = get_params(params)
    κ = sqrt(η' * η)
    p = length(η)
    return log(besselj(0.5p - 1, κ))
end
basemeasure(::Union{<:NaturalParameters{VonMisesFisher}, <:VonMisesFisher}, x) = (1 / 2pi)^(length(x) / 2)

plus(::NaturalParameters{VonMisesFisher}, ::NaturalParameters{VonMisesFisher}) = Plus()