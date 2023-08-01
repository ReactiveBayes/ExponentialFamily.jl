export Poisson

import SpecialFunctions: besseli
import Distributions: Poisson, shape, scale, cov
using DomainSets
using StaticArrays

Distributions.cov(dist::Poisson) = var(dist)

closed_prod_rule(::Type{<:Poisson}, ::Type{<:Poisson}) = ClosedProd()

# NOTE: The product of two Poisson distributions is NOT a Poisson distribution.
function Base.prod(
    ::ClosedProd,
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Poisson}
    η_left = first(getnaturalparameters(left))
    η_right = first(getnaturalparameters(right))

    naturalparameters = η_left + η_right
    basemeasure = (x) -> 1 / factorial(x)^2
    sufficientstatistics = (x) -> SA[x]
    logpartition = (η) -> log(abs(besseli(0, 2 * exp(η / 2))))
    supp = DomainSets.NaturalNumbers()

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

function Base.prod(::ClosedProd, left::Poisson, right::Poisson)
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)

    return prod(ef_left, ef_right)
end

function logpdf_sample_friendly(dist::Poisson)
    λ = params(dist)
    friendly = Poisson(λ)
    return (friendly, friendly)
end

check_valid_natural(::Type{<:Poisson}, params) = isequal(length(params), 1)

pack_naturalparameters(dist::Poisson) = [log(rate(dist))]
function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Poisson})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    return (η1,)
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Poisson) =
    ExponentialFamilyDistribution(Poisson, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Poisson})
    (η,) = unpack_naturalparameters(exponentialfamily)
    return Poisson(exp(η))
end

logpartition(exponentialfamily::ExponentialFamilyDistribution{Poisson}) =
    exp(first(unpack_naturalparameters(exponentialfamily)))

function isproper(exponentialfamily::ExponentialFamilyDistribution{Poisson})
    (η,) = unpack_naturalparameters(exponentialfamily)
    η isa Number && !isnan(η) && !isinf(η)
end

fisherinformation(exponentialfamily::ExponentialFamilyDistribution{Poisson}) =
    SA[exp(first(unpack_naturalparameters(exponentialfamily)));;]

fisherinformation(dist::Poisson) = SA[1 / rate(dist);;]

function support(::ExponentialFamilyDistribution{Poisson})
    return DomainSets.NaturalNumbers()
end

basemeasureconstant(::ExponentialFamilyDistribution{Poisson}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:Poisson}) = NonConstantBaseMeasure()

basemeasure(ef::Union{<:ExponentialFamilyDistribution{Poisson}, <:Poisson}) = x -> basemeasure(ef, x)
function basemeasure(::Union{<:ExponentialFamilyDistribution{Poisson}, <:Poisson}, x::Real)
    return one(x) / factorial(x)
end

sufficientstatistics(ef::ExponentialFamilyDistribution) = (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(::Union{<:ExponentialFamilyDistribution{Poisson}, <:Poisson}, x::Real)
    return SA[x]
end
