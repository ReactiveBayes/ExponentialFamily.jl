export Exponential, ExponentialNaturalParameters

import Distributions: Exponential, params
import SpecialFunctions: digamma, logbeta

vague(::Type{<:Exponential}) = Exponential(1e12)

prod_analytical_rule(::Type{<:Exponential}, ::Type{<:Exponential}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Exponential, right::Exponential)
    θ_left              = left.θ
    θ_right             = right.θ
    return Exponential(inv(inv(θ_left) + inv(θ_right)))
end

function compute_logscale(new_dist::Exponential, left_dist::Exponential, right_dist::Exponential)
    return log(inv(new_dist.θ)) - log(inv(left_dist.θ)) - log(inv(right_dist.θ)) 
end


function mean(::typeof(log), dist::Exponential)
   return -log(rate(dist)) - MathConstants.eulergamma
end

struct ExponentialNaturalParameters{T <: Real} <: NaturalParameters
    rate :: T
end

function naturalparams(dist :: Exponential)
    return ExponentialNaturalParameters(inv(dist.θ))
end

function Distributions.logpdf(dist::ExponentialNaturalParameters, x)
    return log(dist.rate) - dist.rate * x
end