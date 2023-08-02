export Chisq

import SpecialFunctions: loggamma
import Distributions: Chisq, params, dof, var
using StaticArrays

Distributions.cov(dist::Chisq) = var(dist)

closed_prod_rule(::Type{<:Chisq}, ::Type{<:Chisq}) = ClosedProd()

function Base.prod(::ClosedProd, left::Chisq, right::Chisq)
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)

    return prod(ef_left, ef_right)
end

# NOTE: The product of two Chisq distributions is NOT a Chisq distribution.
function Base.prod(
    ::ClosedProd,
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Chisq}
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right
    basemeasure = (x) -> exp(-x)
    sufficientstatistics = (x) -> SA[log(x)]
    logpartition = (η) -> loggamma(η[1] + 1)
    supp = Distributions.support(T)

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

check_valid_natural(::Type{<:Chisq}, params) = length(params) === 1
pack_naturalparameters(dist::Chisq) = [(dof(dist) / 2) - 1]

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Chisq})
    ηvec = getnaturalparameters(ef)
    @inbounds η1 = ηvec[1]

    return (η1,)
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Chisq) =
    ExponentialFamilyDistribution(Chisq, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Chisq})
    (η,) = unpack_naturalparameters(exponentialfamily)
    return Chisq(Int64(2 * (η + one(η))))
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Chisq})
    (η,) = unpack_naturalparameters(exponentialfamily)
    o = one(η)

    return loggamma(η + o) + (η + o) * LOG2
end

function isproper(exponentialfamily::ExponentialFamilyDistribution{Chisq})
    (η,) = unpack_naturalparameters(exponentialfamily)
    return (η > MINUSHALF)
end

struct OpenChi end
struct ClosedChi end

support(ef::ExponentialFamilyDistribution{Chisq}) = support(ef, check_boundaries(ef))
support(::ExponentialFamilyDistribution{Chisq}, ::OpenChi) = OpenInterval{Real}(0, Inf)
support(::ExponentialFamilyDistribution{Chisq}, ::ClosedChi) = ClosedInterval{Real}(0, Inf)
check_boundaries(ef::ExponentialFamilyDistribution{Chisq}) =
    unpack_naturalparameters(ef) == MINUSHALF ? OpenChi() : ClosedChi()

support(dist::Chisq) = support(dist, check_boundaries(dist))
support(::Chisq, ::OpenChi) = OpenInterval{Real}(0, Inf)
support(::Chisq, ::ClosedChi) = ClosedInterval{Real}(0, Inf)
check_boundaries(dist::Chisq) = dof(dist) == 1 ? OpenChi() : ClosedChi()

function fisherinformation(exponentialfamily::ExponentialFamilyDistribution{Chisq})
    (η,) = unpack_naturalparameters(exponentialfamily)
    return SA[trigamma(η + one(η));;]
end

function fisherinformation(dist::Chisq)
    return SA[trigamma(dof(dist) / 2) / 4;;]
end

basemeasureconstant(::ExponentialFamilyDistribution{<:Chisq}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:Chisq}) = NonConstantBaseMeasure()

basemeasure(ef::ExponentialFamilyDistribution{<:Chisq}) = x -> basemeasure(ef, x)
basemeasure(::ExponentialFamilyDistribution{<:Chisq}, x::Real) = exp(-x / 2)

sufficientstatistics(ef::ExponentialFamilyDistribution{<:Chisq}) = x -> sufficientstatistics(ef, x)
sufficientstatistics(::ExponentialFamilyDistribution{<:Chisq}, x::Real) = SA[log(x)]
