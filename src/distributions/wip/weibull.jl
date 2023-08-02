export Weibull

import Distributions: Weibull, params
using DomainSets
using SpecialFunctions: digamma
using HCubature
using StaticArrays

closed_prod_rule(::Type{<:Weibull}, ::Type{<:Weibull}) = ClosedProd()

# NOTE: The product of two Weibull distributions is NOT a Weibull distribution.
function Base.prod(
    ::ClosedProd,
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Weibull}
    conditioner_left = getconditioner(left)
    conditioner_right = getconditioner(right)
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)
    supp = DomainSets.HalfLine()
    if conditioner_left == conditioner_right
        basemeasure = (x) -> x^(2 * (conditioner_left - 1))
        sufficientstatistics = (x) -> SA[x^(conditioner_left)]
        logpartition =
            (η) ->
                log(abs(first(η))^(1 / conditioner_left)) + loggamma(2 - 1 / conditioner_left) -
                2 * log(abs(first(η))) - log(conditioner_left)
        naturalparameters = η_left + η_right

        return ExponentialFamilyDistribution(
            Univariate,
            naturalparameters,
            nothing,
            basemeasure,
            sufficientstatistics,
            logpartition,
            supp
        )
    else
        basemeasure = (x) -> x^(conditioner_left + conditioner_right - 2)
        sufficientstatistics = (x) -> [x^conditioner_left, x^conditioner_right]
        naturalparameters = vcat(η_left, η_right)
        logpartition =
            (η) -> log(
                first(
                    hquadrature(
                        x ->
                            basemeasure(tan(x * pi / 2)) * exp(η' * sufficientstatistics(tan(x * pi / 2))) *
                            (pi / 2) * (1 / cos(x * pi / 2)^2),
                        0,
                        1
                    )
                )
            )

        return ExponentialFamilyDistribution(
            Univariate,
            naturalparameters,
            nothing,
            basemeasure,
            sufficientstatistics,
            logpartition,
            supp
        )
    end
end

function Base.prod(::ClosedProd, left::Weibull, right::Weibull)
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)

    return prod(ClosedProd(), ef_left, ef_right)
end

check_valid_natural(::Type{<:Weibull}, params) = length(params) === 1
check_valid_conditioner(::Type{<:Weibull}, conditioner) = isreal(conditioner) && conditioner > 0

pack_naturalparameters(dist::Weibull) = [-(1 / scale(dist))^(shape(dist))]
function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Weibull})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    return (η1,)
end

function isproper(exponentialfamily::ExponentialFamilyDistribution{Weibull})
    (η,) = unpack_naturalparameters(exponentialfamily)
    return η < 0
end

function basemeasure(dist::Weibull, x)
    return x^(shape(dist) - 1)
end

basemeasureconstant(::ExponentialFamilyDistribution{Weibull}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:Weibull}) = NonConstantBaseMeasure()
basemeasure(ef::ExponentialFamilyDistribution{Weibull}) = basemeasure(ef, x)
function basemeasure(weibull::ExponentialFamilyDistribution{Weibull}, x)
    return x^(getconditioner(weibull) - 1)
end
Base.convert(::Type{ExponentialFamilyDistribution}, dist::Weibull) =
    ExponentialFamilyDistribution(Weibull, pack_naturalparameters(dist), shape(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Weibull})
    k = getconditioner(exponentialfamily)
    (η,) = unpack_naturalparameters(exponentialfamily)
    return Weibull(k, (-1 / η)^(1 / k))
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Weibull})
    return -log(-first(unpack_naturalparameters(exponentialfamily))) - log(getconditioner(exponentialfamily))
end

fisherinformation(exponentialfamily::ExponentialFamilyDistribution{Weibull}) =
    SA[inv(first(unpack_naturalparameters(exponentialfamily)))^2;;]

function fisherinformation(dist::Weibull)
    α = shape(dist)
    θ = scale(dist)

    # see (Fisher Information and the Combination of RGB Channels, Reiner Lenz & Vasileios Zografos, 2013)

    γ = -digamma(1) # Euler-Mascheroni constant (see https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant)
    a11 = (1 - 2γ + γ^2 + π^2 / 6) / (α^2)
    a12 = (γ - 1) / θ
    a21 = a12
    a22 = α^2 / (θ^2)

    return SA[a11 a12; a21 a22]
end

support(::Union{<:ExponentialFamilyDistribution{Weibull}, <:Weibull}) = ClosedInterval{Real}(0, Inf)
insupport(union::Union{<:ExponentialFamilyDistribution{Weibull}, <:Weibull}, x::Real) = x ∈ support(union)

sufficientstatistics(ef::ExponentialFamilyDistribution{Weibull}) = (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(ef::ExponentialFamilyDistribution{Weibull}, x)
    k = getconditioner(ef)
    return SA[x^k]
end

function sufficientstatistics(dist::Weibull, x)
    k = shape(dist)
    return SA[x^k]
end
