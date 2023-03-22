export Exponential

import Distributions: Exponential, params
import SpecialFunctions: digamma, logbeta

vague(::Type{<:Exponential}) = Exponential(1e12)

prod_analytical_rule(::Type{<:Exponential}, ::Type{<:Exponential}) = ClosedProd()

function Base.prod(::ClosedProd, left::Exponential, right::Exponential)
    θ_left  = left.θ
    θ_right = right.θ
    return Exponential(inv(inv(θ_left) + inv(θ_right)))
end

function mean(::typeof(log), dist::Exponential)
    return -log(rate(dist)) - MathConstants.eulergamma
end

function logpartition(dist::Exponential)
    return -log(rate(dist))
end

check_valid_natural(::Type{<:Exponential}, params) = length(params) === 1

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Exponential)
    return KnownExponentialFamilyDistribution(Exponential, [-inv(dist.θ)])
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Exponential})
    return Exponential(-inv(first(getnaturalparameters(exponentialfamily))))
end

function logpartition(η::KnownExponentialFamilyDistribution{Exponential})
    return -log(-first(getnaturalparameters(η)))
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{Exponential}) =
    (first(getnaturalparameters(exponentialfamily)) <= 0)
basemeasure(::Union{<:KnownExponentialFamilyDistribution{Exponential}, <:Exponential}, x) = 1.0
