export NormalMeanVariance

import StatsFuns: log2π, invsqrt2π

"""
    NormalMeanVariance{T <: Real} <: ContinuousUnivariateDistribution

A normal distribution with a known mean `μ` and variance `v`.

# Fields
- `μ::T`: The mean of the normal distribution.
- `v::T`: The variance of the normal distribution.
"""
struct NormalMeanVariance{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    v::T
end

NormalMeanVariance(μ::Real, v::Real)       = NormalMeanVariance(promote(μ, v)...)
NormalMeanVariance(μ::Integer, v::Integer) = NormalMeanVariance(float(μ), float(v))
NormalMeanVariance(μ::T) where {T <: Real} = NormalMeanVariance(μ, one(T))
NormalMeanVariance()                       = NormalMeanVariance(0.0, 1.0)

Distributions.@distr_support NormalMeanVariance -Inf Inf

BayesBase.support(dist::NormalMeanVariance) = Distributions.RealInterval(minimum(dist), maximum(dist))
BayesBase.weightedmean(dist::NormalMeanVariance) = precision(dist) * mean(dist)

function BayesBase.weightedmean_invcov(dist::NormalMeanVariance)
    w = invcov(dist)
    xi = w * mean(dist)
    return (xi, w)
end

BayesBase.mean(dist::NormalMeanVariance)            = dist.μ
BayesBase.median(dist::NormalMeanVariance)          = mean(dist)
BayesBase.mode(dist::NormalMeanVariance)            = mean(dist)
BayesBase.var(dist::NormalMeanVariance)             = dist.v
BayesBase.std(dist::NormalMeanVariance)             = sqrt(var(dist))
BayesBase.cov(dist::NormalMeanVariance)             = var(dist)
BayesBase.invcov(dist::NormalMeanVariance)          = inv(cov(dist))
BayesBase.entropy(dist::NormalMeanVariance)         = (1 + log2π + log(var(dist))) / 2
BayesBase.params(dist::NormalMeanVariance)          = (dist.μ, dist.v)
BayesBase.kurtosis(dist::NormalMeanVariance)        = kurtosis(convert(Normal, dist))
BayesBase.skewness(dist::NormalMeanVariance)        = skewness(convert(Normal, dist))

BayesBase.pdf(dist::NormalMeanVariance, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) / 2cov(dist))) / std(dist)
BayesBase.logpdf(dist::NormalMeanVariance, x::Real) = -(log2π + log(var(dist)) + abs2(x - mean(dist)) / var(dist)) / 2

Base.precision(dist::NormalMeanVariance{T}) where {T} = invcov(dist)
Base.eltype(::NormalMeanVariance{T}) where {T}        = T

Base.convert(::Type{NormalMeanVariance}, μ::Real, v::Real) = NormalMeanVariance(μ, v)
Base.convert(::Type{NormalMeanVariance{T}}, μ::Real, v::Real) where {T <: Real} =
    NormalMeanVariance(convert(T, μ), convert(T, v))

BayesBase.vague(::Type{<:NormalMeanVariance}) = NormalMeanVariance(0.0, huge)
BayesBase.default_prod_rule(::Type{<:NormalMeanVariance}, ::Type{<:NormalMeanVariance}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::NormalMeanVariance, right::NormalMeanVariance)
    xi = mean(left) / var(left) + mean(right) / var(right)
    w = 1 / var(left) + 1 / var(right)
    return NormalWeightedMeanPrecision(xi, w)
end
