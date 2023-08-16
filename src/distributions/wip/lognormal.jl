export LogNormal

import SpecialFunctions: digamma
import Distributions: LogNormal
using StaticArrays

Distributions.cov(dist::LogNormal) = var(dist)

vague(::Type{<:LogNormal}) = LogNormal(1, 1e12)

default_prod_rule(::Type{<:LogNormal}, ::Type{<:LogNormal}) = ClosedProd()

function Base.prod(::ClosedProd, left::LogNormal, right::LogNormal)
    mean1, scale1 = params(left)
    mean2, scale2 = params(right)
    var1 = scale1^2
    var2 = scale2^2

    var3 = (var1 * var2) / (var1 + var2)
    mean3 = var3 * (mean1 / var1 + mean2 / var2 - 1)

    return LogNormal(mean3, sqrt(var3))
end

check_valid_natural(::Type{<:LogNormal}, params) = length(params) === 2

function pack_naturalparameters(dist::LogNormal)
    μ, scale = params(dist)
    var = scale^2

    return [μ / var, -1 / (2 * var)]
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:LogNormal})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    @inbounds η2 = η[2]

    return η1, η2
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::LogNormal) =
    ExponentialFamilyDistribution(LogNormal, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{LogNormal})
    η1, η2 = unpack_naturalparameters(exponentialfamily)
    return LogNormal(-η1 / (2 * η2), sqrt(-1 / (2 * η2)))
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{LogNormal})
    η1, η2 = unpack_naturalparameters(exponentialfamily)
    return -(η1)^2 / (4 * η2) - log(-2η2) / 2
end

function isproper(exponentialfamily::ExponentialFamilyDistribution{LogNormal})
    _, η2 = unpack_naturalparameters(exponentialfamily)
    return (η2 < 0)
end

support(::Union{<:ExponentialFamilyDistribution{LogNormal}, <:LogNormal}) = ClosedInterval{Real}(0, Inf)

basemeasureconstant(::ExponentialFamilyDistribution{LogNormal}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:LogNormal}) = NonConstantBaseMeasure()
basemeasure(ef::ExponentialFamilyDistribution{LogNormal}) = (x) -> basemeasure(ef, x)
function basemeasure(::ExponentialFamilyDistribution{LogNormal}, x::Real)
    return inv(sqrt(TWOPI) * x)
end

function basemeasure(::LogNormal, x::Real)
    return inv(sqrt(TWOPI) * x)
end

function fisherinformation(d::LogNormal)
    σ = d.σ
    return SA[1/(σ^2) 0.0; 0.0 2/(σ^2)]
end

function fisherinformation(ef::ExponentialFamilyDistribution{LogNormal})
    η1, η2 = unpack_naturalparameters(ef)
    return SA[-1/(2η2) (η1)/(2η2^2); (η1)/(2η2^2) -(η1)^2/(2*(η2^3))+1/(2*η2^2)]
end

sufficientstatistics(ef::ExponentialFamilyDistribution{LogNormal}) = (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(::ExponentialFamilyDistribution{LogNormal}, x::Real)
    return SA[log(x), log(x)^2]
end
