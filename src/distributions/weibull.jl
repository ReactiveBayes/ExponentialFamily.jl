export Weibull

import Distributions: Weibull, params
using DomainSets
using SpecialFunctions: digamma
using HCubature

closed_prod_rule(::Type{<:Weibull}, ::Type{<:Weibull}) = ClosedProd()

function Base.prod(
    ::ClosedProd,
    left::KnownExponentialFamilyDistribution{T},
    right::KnownExponentialFamilyDistribution{T}
) where {T <: Weibull}
    conditioner_left = getconditioner(left)
    conditioner_right = getconditioner(right)
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)
    supp = DomainSets.HalfLine()
    if conditioner_left == conditioner_right
        basemeasure = (x) -> x^(2 * (conditioner_left - 1))
        sufficientstatistics = (x) -> x^(conditioner_left)
        logpartition =
            (η) ->
                log(abs(first(η))^(1 / conditioner_left)) + loggamma(2 - 1 / conditioner_left) -
                2 * log(abs(first(η))) - log(conditioner_left)
        naturalparameters = η_left + η_right

        return ExponentialFamilyDistribution(
            Float64,
            basemeasure,
            sufficientstatistics,
            naturalparameters,
            logpartition,
            supp
        )
    else
        basemeasure = (x) -> x^(conditioner_left + conditioner_right - 2)
        sufficientstatistics = (x) -> [x^conditioner_left, x^conditioner_right]
        naturalparameters = [η_left, η_right]
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
            Float64,
            basemeasure,
            sufficientstatistics,
            naturalparameters,
            logpartition,
            supp
        )
    end
end

function Base.prod(::ClosedProd, left::Weibull, right::Weibull)
    ef_left = convert(KnownExponentialFamilyDistribution, left)
    ef_right = convert(KnownExponentialFamilyDistribution, right)

    return prod(ClosedProd(), ef_left, ef_right)
end

check_valid_natural(::Type{<:Weibull}, params) = length(params) === 1
check_valid_conditioner(::Type{<:Weibull}, conditioner) = isreal(conditioner) && conditioner > 0

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    η = getnaturalparameters(exponentialfamily)
    return η < 0
end

function basemeasure(dist::Weibull, x)
    @assert 0 <= x "sufficientstatistics for Weibull should be evaluated at values greater than 0"
    return x^(shape(dist) - 1)
end
function basemeasure(weibull::KnownExponentialFamilyDistribution{Weibull}, x)
    @assert 0 <= x "sufficientstatistics for Weibull should be evaluated at values greater than 0"
    return x^(getconditioner(weibull) - 1)
end
Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Weibull) =
    KnownExponentialFamilyDistribution(Weibull, -(1 / scale(dist))^(shape(dist)), shape(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    k = getconditioner(exponentialfamily)
    η = getnaturalparameters(exponentialfamily)
    return Weibull(k, (-1 / η)^(1 / k))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    return -log(-getnaturalparameters(exponentialfamily)) - log(getconditioner(exponentialfamily))
end

fisherinformation(exponentialfamily::KnownExponentialFamilyDistribution{Weibull}) =
    inv(getnaturalparameters(exponentialfamily))^2

function fisherinformation(dist::Weibull)
    α = shape(dist)
    θ = scale(dist)

    # see (Fisher Information and the Combination of RGB Channels, Reiner Lenz & Vasileios Zografos, 2013)

    γ = -digamma(1) # Euler-Mascheroni constant (see https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant)
    a11 = (1 - 2γ + γ^2 + π^2 / 6) / (α^2)
    a12 = (γ - 1) / θ
    a21 = a12
    a22 = α^2 / (θ^2)

    return [a11 a12; a21 a22]
end

support(::Union{<:KnownExponentialFamilyDistribution{Weibull}, <:Weibull}) = ClosedInterval{Real}(0, Inf)
insupport(union::Union{<:KnownExponentialFamilyDistribution{Weibull}, <:Weibull}, x::Real) = x ∈ support(union)

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Weibull}, x)
    @assert insupport(ef, x) "sufficientstatistics for Weibull should be evaluated at values greater than 0"
    k = getconditioner(ef)
    return x^k
end

function sufficientstatistics(dist::Weibull, x)
    @assert insupport(dist, x) "sufficientstatistics for Weibull should be evaluated at values greater than 0"
    k = shape(dist)
    return x^k
end
