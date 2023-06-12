export Rayleigh

import Distributions: Rayleigh, params
using DomainSets

vague(::Type{<:Rayleigh}) = Rayleigh(Float64(huge))

prod_analytical_rule(::Type{<:Rayleigh}, ::Type{<:Rayleigh}) = ClosedProd()

function Base.prod(
    ::ClosedProd,
    left::KnownExponentialFamilyDistribution{T},
    right::KnownExponentialFamilyDistribution{T}
) where {T <: Rayleigh}
    η1 = first(getnaturalparameters(left))
    η2 = first(getnaturalparameters(right))
    naturalparameters = [η1 + η2]
    basemeasure = (x) -> 4 * x^2 / sqrt(pi)
    sufficientstatistics = (x) -> x^2
    logpartition = (η) -> log(η^(-3 / 2))
    support = DomainSets.HalfLine()

    return ExponentialFamilyDistribution(
        Float64,
        basemeasure,
        sufficientstatistics,
        naturalparameters,
        logpartition,
        support
    )
end

function Base.prod(::ClosedProd, left::Rayleigh, right::Rayleigh)
    σ1 = first(params(left))
    σ2 = first(params(right))
    naturalparameters = [-0.5(σ1^2 + σ2^2) / (σ1 * σ2)^2]
    basemeasure = (x) -> 4 * x^2 / sqrt(pi)
    sufficientstatistics = (x) -> x^2
    logpartition = (η) -> log(η^(-3 / 2))
    support = DomainSets.HalfLine()

    return ExponentialFamilyDistribution(
        Float64,
        basemeasure,
        sufficientstatistics,
        naturalparameters,
        logpartition,
        support
    )
end

function isproper(ef::KnownExponentialFamilyDistribution{Rayleigh})
    η = first(getnaturalparameters(ef))
    return (η < 0)
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Rayleigh)
    σ = first(params(dist))
    KnownExponentialFamilyDistribution(Rayleigh, [-1 / (2σ^2)])
end

function Base.convert(::Type{Distribution}, ef::KnownExponentialFamilyDistribution{Rayleigh})
    η = first(getnaturalparameters(ef))
    return Rayleigh(sqrt(-1 / (2η)))
end

check_valid_natural(::Type{<:Rayleigh}, v) = length(v) === 1

logpartition(ef::KnownExponentialFamilyDistribution{Rayleigh}) = log(-2first(getnaturalparameters(ef)))

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x) = x

fisherinformation(dist::Rayleigh) = 4 / scale(dist)^2

fisherinformation(ef::KnownExponentialFamilyDistribution{Rayleigh}) = -inv(first(getnaturalparameters(ef))^2)

sufficientstatistics(::Union{<:KnownExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x) = x^2