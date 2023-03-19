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

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Exponential)
    return ExponentialFamilyDistribution(Exponential, [-inv(dist.θ)])
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Exponential})
    return Exponential(-inv(first(getnaturalparameters(exponentialfamily))))
end

function lognormalizer(η::ExponentialFamilyDistribution{Exponential})
    return -log(-first(getnaturalparameters(η)))
end

isproper(exponentialfamily::ExponentialFamilyDistribution{Exponential}) = (first(getnaturalparameters(exponentialfamily)) <= 0)
basemeasure(::Union{<:ExponentialFamilyDistribution{Exponential}, <:Exponential}, x) = 1.0
plus(::ExponentialFamilyDistribution{Exponential}, ::ExponentialFamilyDistribution{Exponential}) = Plus()
