export NormalWeightedMeanPrecision

import StatsFuns: log2π, invsqrt2π

"""
    NormalWeightedMeanPrecision{T <: Real} <: ContinuousUnivariateDistribution

A normal distribution parametrized by its natural parameters: the weighted mean `xi` and precision `w`.

# Fields
- `xi::T`: The weighted mean of the normal distribution. `xi` is computed as `w * μ`, where `μ` is the mean of the distribution.
- `w::T`: The precision (inverse variance) of the normal distribution.
"""
struct NormalWeightedMeanPrecision{T <: Real} <: ContinuousUnivariateDistribution
    xi :: T # Weighted mean: xi = w * μ
    w  :: T
end

NormalWeightedMeanPrecision(xi::Real, w::Real)       = NormalWeightedMeanPrecision(promote(xi, w)...)
NormalWeightedMeanPrecision(xi::Integer, w::Integer) = NormalWeightedMeanPrecision(float(xi), float(w))
NormalWeightedMeanPrecision(xi::Real)                = NormalWeightedMeanPrecision(xi, one(xi))
NormalWeightedMeanPrecision()                        = NormalWeightedMeanPrecision(0.0, 1.0)

function NormalWeightedMeanPrecision(xi::T1, w::UniformScaling{T2}) where {T1 <: Real, T2}
    T = promote_type(T1, T2)
    xi_new = convert(T, xi)
    w_new = convert(T, w.λ)
    return NormalWeightedMeanPrecision(xi_new, w_new)
end

Distributions.@distr_support NormalWeightedMeanPrecision -Inf Inf

BayesBase.support(dist::NormalWeightedMeanPrecision) = Distributions.RealInterval(minimum(dist), maximum(dist))

BayesBase.weightedmean(dist::NormalWeightedMeanPrecision) = dist.xi

function BayesBase.mean_cov(dist::NormalWeightedMeanPrecision)
    v = cov(dist)
    μ = v * weightedmean(dist)
    return (μ, v)
end

BayesBase.mean(dist::NormalWeightedMeanPrecision)            = var(dist) * weightedmean(dist)
BayesBase.median(dist::NormalWeightedMeanPrecision)          = mean(dist)
BayesBase.mode(dist::NormalWeightedMeanPrecision)            = mean(dist)
BayesBase.var(dist::NormalWeightedMeanPrecision)             = inv(dist.w)
BayesBase.std(dist::NormalWeightedMeanPrecision)             = sqrt(var(dist))
BayesBase.cov(dist::NormalWeightedMeanPrecision)             = var(dist)
BayesBase.invcov(dist::NormalWeightedMeanPrecision)          = dist.w
BayesBase.entropy(dist::NormalWeightedMeanPrecision)         = (1 + log2π - log(precision(dist))) / 2
BayesBase.params(dist::NormalWeightedMeanPrecision)          = (weightedmean(dist), precision(dist))
BayesBase.kurtosis(dist::NormalWeightedMeanPrecision)        = kurtosis(convert(Normal, dist))
BayesBase.skewness(dist::NormalWeightedMeanPrecision)        = skewness(convert(Normal, dist))
BayesBase.pdf(dist::NormalWeightedMeanPrecision, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) * precision(dist) / 2)) * sqrt(precision(dist))
BayesBase.logpdf(dist::NormalWeightedMeanPrecision, x::Real) = -(log2π - log(precision(dist)) + abs2(x - mean(dist)) * precision(dist)) / 2

Base.precision(dist::NormalWeightedMeanPrecision)       = invcov(dist)
Base.eltype(::NormalWeightedMeanPrecision{T}) where {T} = T

Base.convert(::Type{NormalWeightedMeanPrecision}, xi::Real, w::Real) = NormalWeightedMeanPrecision(xi, w)
Base.convert(::Type{NormalWeightedMeanPrecision{T}}, xi::Real, w::Real) where {T <: Real} =
    NormalWeightedMeanPrecision(convert(T, xi), convert(T, w))

BayesBase.vague(::Type{<:NormalWeightedMeanPrecision}) = NormalWeightedMeanPrecision(0.0, tiny)
BayesBase.default_prod_rule(::Type{<:NormalWeightedMeanPrecision}, ::Type{<:NormalWeightedMeanPrecision}) =
    PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::NormalWeightedMeanPrecision, right::NormalWeightedMeanPrecision)
    xi = weightedmean(left) + weightedmean(right)
    w  = precision(left) + precision(right)
    return NormalWeightedMeanPrecision(xi, w)
end
