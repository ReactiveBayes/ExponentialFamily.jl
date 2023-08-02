export Gamma, GammaShapeScale, GammaDistributionsFamily

import SpecialFunctions: loggamma, digamma
import Distributions: shape, scale, cov
import StatsFuns: log2π
using IntervalSets
using StaticArrays

"""
    GammaShapeScale{T}

A continuous univariate gamma distribution parametrized by its shape `α` and scale `β` parameters.

# Parameters
- `α`: The shape parameter of the gamma distribution. It should be a positive real number.
- `β`: The scale parameter of the gamma distribution. It should be a positive real number.

# Note
 - GammaShapeScale is an alias for Gamma from Distributions.jl.
"""
const GammaShapeScale = Gamma
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

closed_prod_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeScale}) = ClosedProd()

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

closed_prod_rule(::Type{<:GammaShapeRate}, ::Type{<:GammaShapeScale}) = ClosedProd()
closed_prod_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeRate}) = ClosedProd()

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

pack_naturalparameters(dist::GammaDistributionsFamily) = [shape(dist) - one(Float64), -rate(dist)]

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:GammaDistributionsFamily})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    @inbounds η2 = η[2]

    return η1, η2
end

function Base.convert(
    ::Type{Distribution},
    exponentialfamily::ExponentialFamilyDistribution{<:GammaDistributionsFamily}
)
    η1, η2 = unpack_naturalparameters(exponentialfamily)
    return GammaShapeRate(η1 + one(η1), -η2)
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::GammaDistributionsFamily) =
    ExponentialFamilyDistribution(GammaShapeRate, pack_naturalparameters(dist))

function logpartition(exponentialfamily::ExponentialFamilyDistribution{<:GammaDistributionsFamily})
    η1, η2 = unpack_naturalparameters(exponentialfamily)
    return loggamma(η1 + one(η1)) - (η1 + one(η1)) * log(-η2)
end

function isproper(exponentialfamily::ExponentialFamilyDistribution{<:GammaDistributionsFamily})
    a, b = unpack_naturalparameters(exponentialfamily)
    return (a >= tiny - one(a)) && (-b >= tiny)
end

support(::Union{<:ExponentialFamilyDistribution{<:GammaDistributionsFamily}, <:GammaDistributionsFamily}) =
    OpenInterval{Real}(0, Inf)
sufficientstatistics(ef::ExponentialFamilyDistribution{<:GammaDistributionsFamily}) = (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(
    ::ExponentialFamilyDistribution{<:GammaDistributionsFamily},
    x::Real
)
    return SA[log(x), x]
end

basemeasure(::ExponentialFamilyDistribution{<:GammaDistributionsFamily}) = one(Float64)
function basemeasure(
    ::ExponentialFamilyDistribution{<:GammaDistributionsFamily},
    x::Real
)
    return one(x)
end

function fisherinformation(exponentialfamily::ExponentialFamilyDistribution{<:GammaDistributionsFamily})
    η1, η2 = unpack_naturalparameters(exponentialfamily)
    return SA[trigamma(η1 + one(η1)) -one(η2)/η2; -one(η2)/η2 (η1+one(η1))/(η2^2)]
end

function fisherinformation(dist::GammaShapeScale)
    return SA[
        trigamma(shape(dist)) one(scale(dist))/scale(dist)
        one(scale(dist))/scale(dist) shape(dist)/(scale(dist)^2)
    ]
end

function fisherinformation(dist::GammaShapeRate)
    return SA[
        trigamma(shape(dist)) -one(rate(dist))/rate(dist)
        -one(rate(dist))/rate(dist) shape(dist)/(rate(dist)^2)
    ]
end
