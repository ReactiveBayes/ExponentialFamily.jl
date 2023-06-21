export Poisson

import SpecialFunctions: besseli
import Distributions: Poisson, shape, scale, cov
using DomainSets

Distributions.cov(dist::Poisson) = var(dist)

prod_closed_rule(::Type{<:Poisson}, ::Type{<:Poisson}) = ClosedProd()

function Base.prod(
    ::ClosedProd,
    left::KnownExponentialFamilyDistribution{T},
    right::KnownExponentialFamilyDistribution{T}
) where {T <: Poisson}
    η_left = first(getnaturalparameters(left))
    η_right = first(getnaturalparameters(right))

    naturalparameters = η_left + η_right
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

function Base.prod(::ClosedProd, left::Poisson, right::Poisson)
    ef_left = convert(KnownExponentialFamilyDistribution, left)
    ef_right = convert(KnownExponentialFamilyDistribution, right)

    return prod(ef_left, ef_right)
end

function logpdf_sample_friendly(dist::Poisson)
    λ = params(dist)
    friendly = Poisson(λ)
    return (friendly, friendly)
end

check_valid_natural(::Type{<:Poisson}, params) = isequal(length(params), 1)

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Poisson) =
    KnownExponentialFamilyDistribution(Poisson, log(rate(dist)))

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Poisson})
    η = getnaturalparameters(exponentialfamily)
    return Poisson(exp(η))
end

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Poisson}) =
    exp(getnaturalparameters(exponentialfamily))

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Poisson})
    η = getnaturalparameters(exponentialfamily)
    η isa Number && !isnan(η) && !isinf(η)
end

fisherinformation(exponentialfamily::KnownExponentialFamilyDistribution{Poisson}) =
    exp(getnaturalparameters(exponentialfamily))

fisherinformation(dist::Poisson) = 1 / rate(dist)

insupport(::Union{<:KnownExponentialFamilyDistribution{Poisson}, <:Poisson}, x::Real) = typeof(x) <: Integer && 0 <= x

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Poisson}, <:Poisson}, x::Real)
    @assert insupport(union, x) "$(x) is not in the support of Poisson"
    return one(typeof(x)) / factorial(x)
end

function sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{Poisson}, <:Poisson}, x::Real)
    @assert insupport(union, x) "$(x) is not in the support of Poisson"
    return x
end
