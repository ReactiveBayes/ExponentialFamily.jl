export Gamma, GammaShapeScale, GammaDistributionsFamily

import SpecialFunctions: loggamma, digamma
import Distributions: shape, scale, cov
import StatsFuns: log2π
using IntervalSets
using StaticArrays

"""
    GammaShapeScale{T}

A continuous univariate gamma distribution parametrized by its shape `α` and scale `β` parameters.

# Fields
- `α`: The shape parameter of the gamma distribution. It should be a positive real number.
- `β`: The scale parameter of the gamma distribution. It should be a positive real number.

# Note
- GammaShapeScale is an alias for Gamma from Distributions.jl.
"""
const GammaShapeScale = Gamma

function BayesBase.mean(::typeof(log), dist::GammaShapeScale)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

function BayesBase.mean(::typeof(loggamma), dist::GammaShapeScale)
    k, θ = params(dist)
    return 0.5 * (log2π - (digamma(k) + log(θ))) + mean(dist) * (-one(k) + digamma(k + one(k)) + log(θ))
end

function BayesBase.mean(::typeof(xtlog), dist::GammaShapeScale)
    k, θ = params(dist)
    return mean(dist) * (digamma(k + one(k)) + log(θ))
end

BayesBase.vague(::Type{<:GammaShapeScale}) = GammaShapeScale(one(Float64), huge)
BayesBase.default_prod_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeScale}) = ClosedProd()

function BayesBase.prod(::ClosedProd, left::GammaShapeScale, right::GammaShapeScale)
    return GammaShapeScale(
        shape(left) + shape(right) - 1,
        (scale(left) * scale(right)) / (scale(left) + scale(right))
    )
end
