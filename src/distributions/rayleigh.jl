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
    η1 = getnaturalparameters(left)
    η2 = getnaturalparameters(right)
    naturalparameters = η1 + η2
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

function Base.prod(::ClosedProd, left::T, right::T) where {T <: Rayleigh}
    ef_left = convert(KnownExponentialFamilyDistribution, left)
    ef_right = convert(KnownExponentialFamilyDistribution, right)
    return prod(ClosedProd(), ef_left, ef_right)
end

function isproper(ef::KnownExponentialFamilyDistribution{Rayleigh})
    η = getnaturalparameters(ef)
    return (η < 0)
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Rayleigh)
    σ = first(params(dist))
    KnownExponentialFamilyDistribution(Rayleigh, -1 / (2 * σ^2))
end

function Base.convert(::Type{Distribution}, ef::KnownExponentialFamilyDistribution{Rayleigh})
    η = getnaturalparameters(ef)
    return Rayleigh(sqrt(-1 / (2η)))
end

check_valid_natural(::Type{<:Rayleigh}, v) = length(v) === 1

logpartition(ef::KnownExponentialFamilyDistribution{Rayleigh}) = -log(-2 * getnaturalparameters(ef))

fisherinformation(dist::Rayleigh) = 4 / scale(dist)^2

fisherinformation(ef::KnownExponentialFamilyDistribution{Rayleigh}) = inv(getnaturalparameters(ef)^2)

support(::KnownExponentialFamilyDistribution{Rayleigh}) = ClosedInterval{Real}(0, Inf)

function sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x::Real)
    @assert insupport(union, x) "Rayleigh sufficient statistics should be evaluated at values greater than 0"
    return x^2
end

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x::Real)
    @assert insupport(union, x) "Rayleigh base measure should be evaluated at values greater than 0"
    return x
end
