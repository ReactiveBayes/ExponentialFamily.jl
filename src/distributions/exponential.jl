export Exponential

import Distributions: Exponential, params
import SpecialFunctions: digamma, logbeta

vague(::Type{<:Exponential}) = Exponential(1e12)

prod_analytical_rule(::Type{<:Exponential}, ::Type{<:Exponential}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Exponential, right::Exponential)
    θ_left  = left.θ
    θ_right = right.θ
    return Exponential(inv(inv(θ_left) + inv(θ_right)))
end

function mean(::typeof(log), dist::Exponential)
    return -log(rate(dist)) - MathConstants.eulergamma
end

function lognormalizer(dist::Exponential)
    return -log(rate(dist))
end

check_valid_natural(::Type{<:Exponential}, params) = length(params) === 1

function Base.convert(::Type{NaturalParameters}, dist::Exponential)
    return NaturalParameters(Exponential, [-inv(dist.θ)])
end

function Base.convert(::Type{Distribution}, params::NaturalParameters{Exponential})
    return Exponential(-inv(first(get_params(params))))
end

function lognormalizer(η::NaturalParameters{Exponential})
    return -log(-first(get_params(η)))
end

isproper(::Type{<:Exponential}, parms) = (first(params) <= 0)
basemeasure(::Union{<:NaturalParameters{Exponential}, <:Exponential}, x) = 1.0
plus(::NaturalParameters{Exponential}, ::NaturalParameters{Exponential}) = Plus()