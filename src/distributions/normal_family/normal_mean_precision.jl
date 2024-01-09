export NormalMeanPrecision

import StatsFuns: log2π, invsqrt2π

"""
    NormalMeanPrecision{T <: Real} <: ContinuousUnivariateDistribution

A normal distribution with a known mean `μ` and precision `w`.

# Fields
- `μ::T`: The mean of the normal distribution.
- `w::T`: The precision of the normal distribution.
"""
struct NormalMeanPrecision{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    w::T
end

NormalMeanPrecision(μ::Real, w::Real)       = NormalMeanPrecision(promote(μ, w)...)
NormalMeanPrecision(μ::Integer, w::Integer) = NormalMeanPrecision(float(μ), float(w))
NormalMeanPrecision(μ::Real)                = NormalMeanPrecision(μ, one(μ))
NormalMeanPrecision()                       = NormalMeanPrecision(0.0, 1.0)

Distributions.@distr_support NormalMeanPrecision -Inf Inf

BayesBase.support(dist::NormalMeanPrecision) = Distributions.RealInterval(minimum(dist), maximum(dist))

BayesBase.weightedmean(dist::NormalMeanPrecision) = precision(dist) * mean(dist)

BayesBase.mean(dist::NormalMeanPrecision) = dist.μ
BayesBase.median(dist::NormalMeanPrecision) = mean(dist)
BayesBase.mode(dist::NormalMeanPrecision) = mean(dist)
BayesBase.var(dist::NormalMeanPrecision) = inv(dist.w)
BayesBase.std(dist::NormalMeanPrecision) = sqrt(var(dist))
BayesBase.cov(dist::NormalMeanPrecision) = var(dist)
BayesBase.invcov(dist::NormalMeanPrecision) = dist.w
BayesBase.entropy(dist::NormalMeanPrecision) = (1 + log2π - log(precision(dist))) / 2
BayesBase.params(dist::NormalMeanPrecision) = (mean(dist), precision(dist))
BayesBase.kurtosis(dist::NormalMeanPrecision) = kurtosis(convert(Normal, dist))
BayesBase.skewness(dist::NormalMeanPrecision) = skewness(convert(Normal, dist))

BayesBase.pdf(dist::NormalMeanPrecision, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) * precision(dist) / 2)) * sqrt(precision(dist))
BayesBase.logpdf(dist::NormalMeanPrecision, x::Real) = -(log2π - log(precision(dist)) + abs2(x - mean(dist)) * precision(dist)) / 2

Base.precision(dist::NormalMeanPrecision)       = invcov(dist)
Base.eltype(::NormalMeanPrecision{T}) where {T} = T

Base.convert(::Type{NormalMeanPrecision}, μ::Real, w::Real) = NormalMeanPrecision(μ, w)
Base.convert(::Type{NormalMeanPrecision{T}}, μ::Real, w::Real) where {T <: Real} =
    NormalMeanPrecision(convert(T, μ), convert(T, w))

BayesBase.vague(::Type{<:NormalMeanPrecision}) = NormalMeanPrecision(0.0, tiny)
BayesBase.default_prod_rule(::Type{<:NormalMeanPrecision}, ::Type{<:NormalMeanPrecision}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::NormalMeanPrecision, right::NormalMeanPrecision)
    w = precision(left) + precision(right)
    xi = mean(left) * precision(left) + mean(right) * precision(right)
    return NormalWeightedMeanPrecision(xi, w)
end
