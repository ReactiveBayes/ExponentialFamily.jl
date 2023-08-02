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

Distributions.support(dist::NormalMeanPrecision) = Distributions.RealInterval(minimum(dist), maximum(dist))

weightedmean(dist::NormalMeanPrecision) = precision(dist) * mean(dist)

Distributions.mean(dist::NormalMeanPrecision)    = dist.μ
Distributions.median(dist::NormalMeanPrecision)  = mean(dist)
Distributions.mode(dist::NormalMeanPrecision)    = mean(dist)
Distributions.var(dist::NormalMeanPrecision)     = inv(dist.w)
Distributions.std(dist::NormalMeanPrecision)     = sqrt(var(dist))
Distributions.cov(dist::NormalMeanPrecision)     = var(dist)
Distributions.invcov(dist::NormalMeanPrecision)  = dist.w
Distributions.entropy(dist::NormalMeanPrecision) = (1 + log2π - log(precision(dist))) / 2
Distributions.params(dist::NormalMeanPrecision)  = (mean(dist), precision(dist))

Distributions.pdf(dist::NormalMeanPrecision, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) * precision(dist) / 2)) * sqrt(precision(dist))
Distributions.logpdf(dist::NormalMeanPrecision, x::Real) = -(log2π - log(precision(dist)) + abs2(x - mean(dist)) * precision(dist)) / 2

Base.precision(dist::NormalMeanPrecision)       = invcov(dist)
Base.eltype(::NormalMeanPrecision{T}) where {T} = T

Base.convert(::Type{NormalMeanPrecision}, μ::Real, w::Real) = NormalMeanPrecision(μ, w)
Base.convert(::Type{NormalMeanPrecision{T}}, μ::Real, w::Real) where {T <: Real} =
    NormalMeanPrecision(convert(T, μ), convert(T, w))

vague(::Type{<:NormalMeanPrecision}) = NormalMeanPrecision(0.0, tiny)

default_prod_rule(::Type{<:NormalMeanPrecision}, ::Type{<:NormalMeanPrecision}) = ClosedProd()

function Base.prod(::ClosedProd, left::NormalMeanPrecision, right::NormalMeanPrecision)
    w = precision(left) + precision(right)
    xi = mean(left) * precision(left) + mean(right) * precision(right)
    return NormalWeightedMeanPrecision(xi, w)
end

function fisherinformation(dist::NormalMeanPrecision)
    _, w = params(dist)
    return [w 0; 0 1/(2*w^2)]
end
