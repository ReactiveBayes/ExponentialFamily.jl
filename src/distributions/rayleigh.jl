export Rayleigh

import Distributions: Rayleigh, params
using DomainSets
using StaticArrays

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
    sufficientstatistics = (x) -> SA[x^2]
    logpartition = (η) -> (-3 / 2)log(η)
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

pack_naturalparameters(dist::Rayleigh) = [MINUSHALF / first(params(dist))^2]
function unpack_naturalparameters(ef::KnownExponentialFamilyDistribution{<:Rayleigh})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    return η1
end

isproper(ef::KnownExponentialFamilyDistribution{Rayleigh}) = unpack_naturalparameters(ef) < 0

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Rayleigh) = KnownExponentialFamilyDistribution(Rayleigh, pack_naturalparameters(dist))
    
function Base.convert(::Type{Distribution}, ef::KnownExponentialFamilyDistribution{Rayleigh})
    η = unpack_naturalparameters(ef)
    return Rayleigh(sqrt(-1 / (2η)))
end

check_valid_natural(::Type{<:Rayleigh}, v) = length(v) === 1

logpartition(ef::KnownExponentialFamilyDistribution{Rayleigh}) = -log(-2 * unpack_naturalparameters(ef))

fisherinformation(dist::Rayleigh) = [4 / scale(dist)^2]

fisherinformation(ef::KnownExponentialFamilyDistribution{Rayleigh}) = [inv(unpack_naturalparameters(ef)^2)]

support(::KnownExponentialFamilyDistribution{Rayleigh}) = ClosedInterval{Real}(0, Inf)

function sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x::Real)
    @assert insupport(union, x) "Rayleigh sufficient statistics should be evaluated at values greater than 0"
    return SA[x^2]
end

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x::Real)
    @assert insupport(union, x) "Rayleigh base measure should be evaluated at values greater than 0"
    return x
end
