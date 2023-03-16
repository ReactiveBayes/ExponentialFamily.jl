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
    return 0.5 * (log2π - (digamma(k) + log(θ))) + mean(dist) * (-1 + digamma(k + 1) + log(θ))
end

function mean(::typeof(xtlog), dist::GammaShapeScale)
    k, θ = params(dist)
    return mean(dist) * (digamma(k + 1) + log(θ))
end

vague(::Type{<:GammaShapeScale}) = GammaShapeScale(1.0, huge)

prod_analytical_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeScale}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::GammaShapeScale, right::GammaShapeScale)
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

prod_analytical_rule(::Type{<:GammaShapeRate}, ::Type{<:GammaShapeScale}) = ProdAnalyticalRuleAvailable()
prod_analytical_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeRate}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::GammaShapeRate, right::GammaShapeScale)
    T = promote_samplefloattype(left, right)
    return GammaShapeRate(shape(left) + shape(right) - one(T), rate(left) + rate(right))
end

function Base.prod(::ProdAnalytical, left::GammaShapeScale, right::GammaShapeRate)
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

function logpdf_sample_friendly(dist::GammaDistributionsFamily)
    friendly = convert(GammaShapeScale, dist)
    return (friendly, friendly)
end

check_valid_natural(::Type{<:GammaDistributionsFamily}, params) = (length(params) === 2)

function Base.convert(::Type{Distribution}, params::NaturalParameters{<:GammaDistributionsFamily})
    η = get_params(params)
    η1 = first(η)
    η2 = getindex(η, 2)
    return GammaShapeRate(η1 + 1, -η2)
end

Base.convert(::Type{NaturalParameters}, dist::GammaDistributionsFamily) =
    NaturalParameters(GammaShapeRate, [shape(dist) - 1, -rate(dist)])

function lognormalizer(params::NaturalParameters{<:GammaDistributionsFamily})
    η = get_params(params)
    a = first(η)
    b = getindex(η, 2)
    return loggamma(a + 1) - (a + 1) * log(-b)
end

function isproper(params::NaturalParameters{<:GammaDistributionsFamily})
    η = get_params(params)
    a = first(η)
    b = getindex(η, 2)
    return (a >= tiny - 1) && (-b >= tiny)
end

basemeasure(::Union{<:NaturalParameters{GammaDistributionsFamily}, <:GammaDistributionsFamily}, x) = 1.0
plus(::NaturalParameters{GammaShapeRate}, ::NaturalParameters{GammaShapeRate}) = Plus()
