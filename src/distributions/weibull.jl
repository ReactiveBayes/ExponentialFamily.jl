export Weibull

import Distributions: Weibull, params
using DomainSets
using SpecialFunctions: loggamma
using HCubature

prod_closed_rule(::Type{<:Weibull}, ::Type{<:Weibull}) = ClosedProd()

function Base.prod(::ClosedProd, left::Weibull, right::Weibull)
    ef_left = convert(KnownExponentialFamilyDistribution, left)
    ef_right = convert(KnownExponentialFamilyDistribution, right)
    conditioner_left = getconditioner(ef_left)
    conditioner_right = getconditioner(ef_right)
    η_left = getnaturalparameters(ef_left)
    η_right = getnaturalparameters(ef_right)
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
        naturalparameters = [first(η_left), first(η_right)]
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

check_valid_natural(::Type{<:Weibull}, params) = length(params) === 1
check_valid_conditioner(::Type{<:Weibull}, conditioner) = isreal(conditioner) && conditioner > 0

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    η = getnaturalparameters(exponentialfamily)
    return first(η) < 0
end

basemeasure(dist::Weibull, x) = x^(shape(dist) - 1)
basemeasure(weibull::KnownExponentialFamilyDistribution{Weibull}, x) = x^(getconditioner(weibull) - 1)

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Weibull) =
    KnownExponentialFamilyDistribution(Weibull, [-(1 / scale(dist))^(shape(dist))], shape(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    k = getconditioner(exponentialfamily)
    η = first(getnaturalparameters(exponentialfamily))
    return Weibull(k, (-1 / η)^(1 / k))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    return -log(-first(getnaturalparameters(exponentialfamily))) - log(getconditioner(exponentialfamily))
end
