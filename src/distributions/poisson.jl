export Poisson

import SpecialFunctions: besseli
import Distributions: Poisson, shape, scale, cov
using DomainSets

Distributions.cov(dist::Poisson) = var(dist)

prod_closed_rule(::Type{<:Poisson}, ::Type{<:Poisson}) = ClosedProd()

function Base.prod(::ClosedProd, left::Poisson, right::Poisson)
    η_left = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, left)))
    η_right = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, right)))

    naturalparameters = [η_left + η_right]
    basemeasure = (x) -> 1 / factorial(x)^2
    sufficientstatistics = (x) -> x
    logpartition = (η) -> log(abs(besseli(0, 2 * exp(η / 2))))
    supp = DomainSets.NaturalNumbers()

    return ExponentialFamilyDistribution(
        Float64,
        basemeasure,
        sufficientstatistics,
        naturalparameters,
        logpartition,
        supp
    )
end

function logpdf_sample_friendly(dist::Poisson)
    λ = params(dist)
    friendly = Poisson(λ)
    return (friendly, friendly)
end

check_valid_natural(::Type{<:Poisson}, params) = isequal(length(params), 1)

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Poisson) =
    KnownExponentialFamilyDistribution(Poisson, [log(rate(dist))])

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Poisson})
    η = first(getnaturalparameters(exponentialfamily))
    return Poisson(exp(η))
end

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Poisson}) =
    exp(first(getnaturalparameters(exponentialfamily)))

isproper(exponentialfamily::KnownExponentialFamilyDistribution{Poisson}) =
    first(getnaturalparameters(exponentialfamily)) >= 0

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Poisson}, <:Poisson}, x) = 1.0 / factorial(x)

fisher_information(exponentialfamily::KnownExponentialFamilyDistribution{Poisson}) =
    exp(first(getnaturalparameters(exponentialfamily)))

fisher_information(dist::Poisson) = 1 / rate(dist)
