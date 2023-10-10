export GammaShapeRate

import Distributions: Gamma, shape, rate, logpdf
import SpecialFunctions: loggamma, digamma, gamma
import StatsFuns: log2π
import Random: rand

"""
    GammaShapeRate{T <: Real}

A univariate gamma distribution parametrized by its shape `a` and rate `b`.

# Fields
- `a`: The shape parameter of the gamma distribution. It should be a positive real number.
- `b`: The rate parameter of the gamma distribution. It should be a positive real number.
"""
struct GammaShapeRate{T <: Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
end

GammaShapeRate(a::Real, b::Real)       = GammaShapeRate(promote(a, b)...)
GammaShapeRate(a::Integer, b::Integer) = GammaShapeRate(float(a), float(b))
GammaShapeRate(a::Real)                = GammaShapeRate(a, one(a))
GammaShapeRate()                       = GammaShapeRate(1.0, 1.0)

Distributions.@distr_support GammaShapeRate 0 Inf

BayesBase.support(dist::GammaShapeRate) = Distributions.RealInterval(minimum(dist), maximum(dist))
BayesBase.shape(dist::GammaShapeRate) = dist.a
BayesBase.rate(dist::GammaShapeRate) = dist.b
BayesBase.scale(dist::GammaShapeRate) = inv(dist.b)
BayesBase.mean(dist::GammaShapeRate) = shape(dist) / rate(dist)
BayesBase.var(dist::GammaShapeRate) = shape(dist) / abs2(rate(dist))
BayesBase.params(dist::GammaShapeRate) = (shape(dist), rate(dist))

BayesBase.mode(d::GammaShapeRate) =
    shape(d) >= 1 ? mode(Gamma(shape(d), scale(d))) : throw(error("Gamma has no mode when shape < 1"))

function BayesBase.entropy(dist::GammaShapeRate)
    a, b = params(dist)
    return a - log(b) + loggamma(a) + (1 - a) * digamma(a)
end

function BayesBase.mean(::typeof(log), dist::GammaShapeRate)
    a, b = params(dist)
    return digamma(a) - log(b)
end

function BayesBase.mean(::typeof(loggamma), dist::GammaShapeRate)
    a, b = params(dist)
    return 0.5 * (log2π - (digamma(a) - log(b))) + mean(dist) * (-1 + digamma(a + 1) - log(b))
end

function BayesBase.mean(::typeof(xtlog), dist::GammaShapeRate)
    a, b = params(dist)
    return mean(dist) * (digamma(a + 1) - log(b))
end

Base.eltype(::GammaShapeRate{T}) where {T} = T

Base.convert(::Type{GammaShapeRate{T}}, a::Real, b::Real) where {T <: Real} =
    GammaShapeRate(convert(T, a), convert(T, b))

BayesBase.vague(::Type{<:GammaShapeRate}) = GammaShapeRate(1.0, tiny)
BayesBase.default_prod_rule(::Type{<:GammaShapeRate}, ::Type{<:GammaShapeRate}) = ClosedProd()

function BayesBase.prod(::ClosedProd, left::GammaShapeRate, right::GammaShapeRate)
    T = promote_samplefloattype(left, right)
    return GammaShapeRate(shape(left) + shape(right) - one(T), rate(left) + rate(right))
end

BayesBase.pdf(dist::GammaShapeRate, x::Real)    = exp(logpdf(dist, x))
BayesBase.logpdf(dist::GammaShapeRate, x::Real) = shape(dist) * log(rate(dist)) - loggamma(shape(dist)) + (shape(dist) - 1) * log(x) - rate(dist) * x

function BayesBase.rand(rng::AbstractRNG, dist::GammaShapeRate)
    return convert(eltype(dist), rand(rng, convert(GammaShapeScale, dist)))
end

function BayesBase.rand(rng::AbstractRNG, dist::GammaShapeRate, n::Int64)
    return convert(AbstractArray{eltype(dist)}, rand(rng, convert(GammaShapeScale, dist), n))
end

function BayesBase.rand!(rng::AbstractRNG, dist::GammaShapeRate, container::AbstractVector)
    return rand!(rng, convert(GammaShapeScale, dist), container)
end
