export Gamma, GammaShapeScale, GammaDistributionsFamily

import SpecialFunctions: loggamma, digamma
import Distributions: shape, scale, cov
import StatsFuns: log2π

const GammaShapeScale             = Gamma
const GammaDistributionsFamily{T} = Union{GammaShapeScale{T}, GammaShapeRate{T}}

Distributions.cov(dist::GammaDistributionsFamily) = var(dist)

function mean(::typeof(log), dist::GammaShapeScale)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

function mean(::typeof(loggamma), dist::GammaShapeScale)
    k, θ = params(dist)
    return 0.5 * (log2π - (digamma(k) + log(θ))) + mean(dist) * (-one(k) + digamma(k + one(k)) + log(θ))
end

function mean(::typeof(xtlog), dist::GammaShapeScale)
    k, θ = params(dist)
    return mean(dist) * (digamma(k + one(k)) + log(θ))
end

vague(::Type{<:GammaShapeScale}) = GammaShapeScale(one(Float64), huge)

prod_closed_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeScale}) = ClosedProd()

function Base.prod(::ClosedProd, left::GammaShapeScale, right::GammaShapeScale)
    T = promote_samplefloattype(left, right)
    return GammaShapeScale(
        shape(left) + shape(right) - one(T),
        (scale(left) * scale(right)) / (scale(left) + scale(right))
    )
end

function Base.convert(::Type{GammaShapeScale{T}}, dist::GammaDistributionsFamily) where {T}
    return GammaShapeScale(convert(T, shape(dist)), convert(T, scale(dist)))
end

function Base.convert(::Type{GammaShapeScale}, dist::GammaDistributionsFamily{T}) where {T}
    return convert(GammaShapeScale{T}, dist)
end

function Base.convert(::Type{GammaShapeRate{T}}, dist::GammaDistributionsFamily) where {T}
    return GammaShapeRate(convert(T, shape(dist)), convert(T, rate(dist)))
end

function Base.convert(::Type{GammaShapeRate}, dist::GammaDistributionsFamily{T}) where {T}
    return convert(GammaShapeRate{T}, dist)
end

prod_closed_rule(::Type{<:GammaShapeRate}, ::Type{<:GammaShapeScale}) = ClosedProd()
prod_closed_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeRate}) = ClosedProd()

function Base.prod(::ClosedProd, left::GammaShapeRate, right::GammaShapeScale)
    T = promote_samplefloattype(left, right)
    return GammaShapeRate(shape(left) + shape(right) - one(T), rate(left) + rate(right))
end

function Base.prod(::ClosedProd, left::GammaShapeScale, right::GammaShapeRate)
    T = promote_samplefloattype(left, right)
    return GammaShapeScale(
        shape(left) + shape(right) - one(T),
        (scale(left) * scale(right)) / (scale(left) + scale(right))
    )
end

function compute_logscale(
    new_dist::GammaDistributionsFamily,
    left_dist::GammaDistributionsFamily,
    right_dist::GammaDistributionsFamily
)
    ay, by = shape(new_dist), rate(new_dist)
    ax, bx = shape(left_dist), rate(left_dist)
    az, bz = shape(right_dist), rate(right_dist)
    return loggamma(ay) - loggamma(ax) - loggamma(az) + ax * log(bx) + az * log(bz) - ay * log(by)
end

function logpdf_sample_optimized(dist::GammaDistributionsFamily)
    optimized_dist = convert(GammaShapeScale, dist)
    return (optimized_dist, optimized_dist)
end

check_valid_natural(::Type{<:GammaDistributionsFamily}, params) = (length(params) === 2)

function Base.convert(
    ::Type{Distribution},
    exponentialfamily::KnownExponentialFamilyDistribution{<:GammaDistributionsFamily}
)
    η = getnaturalparameters(exponentialfamily)
    η1 = first(η)
    η2 = getindex(η, 2)
    return GammaShapeRate(η1 + one(η1), -η2)
end

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::GammaDistributionsFamily) =
    KnownExponentialFamilyDistribution(GammaShapeRate, [shape(dist) - one(Float64), -rate(dist)])

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{<:GammaDistributionsFamily})
    η = getnaturalparameters(exponentialfamily)
    a = first(η)
    b = getindex(η, 2)
    return loggamma(a + one(a)) - (a + one(a)) * log(-b)
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{<:GammaDistributionsFamily})
    η = getnaturalparameters(exponentialfamily)
    a = first(η)
    b = getindex(η, 2)
    return (a >= tiny - one(a)) && (-b >= tiny)
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{GammaDistributionsFamily}, <:GammaDistributionsFamily}, x) =
    1.0
