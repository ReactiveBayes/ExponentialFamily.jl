export Chisq

import SpecialFunctions: loggamma
import Distributions: Chisq, params, dof, var

Distributions.cov(dist::Chisq) = var(dist)

prod_closed_rule(::Type{<:Chisq}, ::Type{<:Chisq}) = ClosedProd()

function Base.prod(::ClosedProd, left::Chisq, right::Chisq)
    η_left = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, left)))
    η_right = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, right)))

    naturalparameters = [η_left + η_right]
    basemeasure = (x) -> exp(-x)
    sufficientstatistics = (x) -> log(x)
    logpartition = (η) -> loggamma(η + 1)
    supp = support(left)

    return ExponentialFamilyDistribution(
        Float64,
        basemeasure,
        sufficientstatistics,
        naturalparameters,
        logpartition,
        supp
    )
end

check_valid_natural(::Type{<:Chisq}, params) = length(params) === 1

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Chisq) =
    KnownExponentialFamilyDistribution(Chisq, [dof(dist) / 2 - 1])

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = first(getnaturalparameters(exponentialfamily))
    return Chisq(Int64(2 * (η + 1)))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = first(getnaturalparameters(exponentialfamily))

    return loggamma(η + 1) + (η + 1) * log(2)
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = first(getnaturalparameters(exponentialfamily))

    return (η > -1 / 2)
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Chisq}, <:Chisq}, x) = exp(-x / 2)
