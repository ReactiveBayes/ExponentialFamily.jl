export Exponential, ExponentialNaturalParameters

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

struct ExponentialNaturalParameters{T <: Real} <: NaturalParameters
    minus_rate::T
end

function naturalparams(dist::Exponential)
    return ExponentialNaturalParameters(-inv(dist.θ))
end

function Distributions.logpdf(dist::ExponentialNaturalParameters, x)
    return log(-dist.minus_rate) + dist.minus_rate * x
end

function Base.:+(left::ExponentialNaturalParameters, right::ExponentialNaturalParameters)
    return ExponentialNaturalParameters(left.minus_rate + right.minus_rate)
end

function Base.:-(left::ExponentialNaturalParameters, right::ExponentialNaturalParameters)
    return ExponentialNaturalParameters(left.minus_rate - right.minus_rate)
end

function lognormalizer(η::ExponentialNaturalParameters)
    return -log(-η.minus_rate)
end
