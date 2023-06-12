export Chisq

import SpecialFunctions: loggamma
import Distributions: Chisq, params, dof, var

Distributions.cov(dist::Chisq) = var(dist)

prod_closed_rule(::Type{<:Chisq}, ::Type{<:Chisq}) = ClosedProd()

Distributions.support(::KnownExponentialFamilyDistribution{Chisq}) = support(Chisq)

function Base.prod(::ClosedProd, left::Chisq, right::Chisq)
    ef_left = convert(KnownExponentialFamilyDistribution, left)
    ef_right = convert(KnownExponentialFamilyDistribution, right)

    return prod(ef_left, ef_right)
end

function Base.prod(
    ::ClosedProd,
    left::KnownExponentialFamilyDistribution{T},
    right::KnownExponentialFamilyDistribution{T}
) where {T <: Chisq}
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right
    basemeasure = (x) -> exp(-x)
    sufficientstatistics = (x) -> log(x)
    logpartition = (η) -> loggamma(η + 1)
    supp = support(T)

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
    KnownExponentialFamilyDistribution(Chisq, dof(dist) / 2 - 1)

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = getnaturalparameters(exponentialfamily)
    return Chisq(Int64(2 * (η + 1)))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = getnaturalparameters(exponentialfamily)

    return loggamma(η + 1) + (η + 1) * log(2)
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = getnaturalparameters(exponentialfamily)
    return (η > -1 / 2)
end

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Chisq}, <:Chisq}, x) 
    @assert x < Base.maximum(support(union)) && x > Base.minimum(support(union)) "$(x) is not in the support"
    return exp(-x / 2)
end
function fisherinformation(exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = getnaturalparameters(exponentialfamily)
    return trigamma(η + 1)
end

function fisherinformation(dist::Chisq)
    return trigamma(dof(dist) / 2) / 4
end

function sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{Chisq}, <:Chisq}, x) 
    @assert x < Base.maximum(support(union)) && x > Base.minimum(support(union)) "$(x) is not in the support"
    return log(x)
end