export Pareto

import Distributions: Pareto, shape, scale, params
using StaticArrays

vague(::Type{<:Pareto}) = Pareto(1e12)

Distributions.cov(dist::Type{<:Pareto}) = var(dist)

closed_prod_rule(::Type{<:Pareto}, ::Type{<:Pareto}) = ClosedProd()
pack_naturalparameters(dist::Pareto) = [-shape(dist) - 1]
function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Pareto})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    return (η1,)
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Pareto) =
    ExponentialFamilyDistribution(Pareto, pack_naturalparameters(dist), scale(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{<:Pareto})
    (η,) = unpack_naturalparameters(exponentialfamily)
    conditioner = getconditioner(exponentialfamily)
    return Pareto(-1 - η, conditioner)
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Pareto})
    (η,) = unpack_naturalparameters(exponentialfamily)
    k = getconditioner(exponentialfamily)
    return log(k^(one(η) + η) / (-one(η) - η))
end

check_valid_natural(::Type{<:Pareto}, params) = (length(params) === 1)
check_valid_conditioner(::Type{<:Pareto}, conditioner) = isinteger(conditioner) && conditioner > 0
function isproper(exponentialfamily::ExponentialFamilyDistribution{Pareto})
    (η,) = unpack_naturalparameters(exponentialfamily)
    return (η <= -1)
end

function fisherinformation(ef::ExponentialFamilyDistribution{Pareto})
    (η,) = unpack_naturalparameters(ef)
    return SA[1 / (-1 - η)^2;;]
end

function fisherinformation(dist::Pareto)
    α = shape(dist)
    x = scale(dist)

    return SA[1/α^2 -1/x; -1/x α/x^2]
end

function support(ef::ExponentialFamilyDistribution{Pareto})
    return ClosedInterval{Real}(getconditioner(ef), Inf)
end

sufficientstatistics(ef::ExponentialFamilyDistribution{Pareto}) = (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(::ExponentialFamilyDistribution{Pareto}, x::Real)
    return SA[log(x)]
end

basemeasure(::ExponentialFamilyDistribution{Pareto}) = one(Float64)
function basemeasure(::ExponentialFamilyDistribution{Pareto}, x::Real)
    return one(x)
end
