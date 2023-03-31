export Rayleigh

import Distributions: Rayleigh, params

vague(::Type{<:Rayleigh}) = Rayleigh(Float64(huge))

prod_analytical_rule(::Type{<:Rayleigh}, ::Type{<:Rayleigh}) = ClosedProd()

function Base.prod(::ClosedProd, left::Rayleigh, right::Rayleigh)
    varleft = first(params(left))^2
    varright = first(params(right))^2

    return Rayleigh(sqrt(varleft * varright / (varleft + varright)))
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
