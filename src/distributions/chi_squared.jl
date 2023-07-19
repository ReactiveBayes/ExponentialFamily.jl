export Chisq

import SpecialFunctions: loggamma
import Distributions: Chisq, params, dof, var
const log2 = log(2)
const minushalf = - 1/2

Distributions.cov(dist::Chisq) = var(dist)

closed_prod_rule(::Type{<:Chisq}, ::Type{<:Chisq}) = ClosedProd()

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
    logpartition = (η) -> loggamma(η[1] + 1)
    supp = Distributions.support(T)

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
pack_naturalparameters(dist::Chisq) = [(dof(dist)/2) - 1]

function unpack_naturalparameters(ef::KnownExponentialFamilyDistribution{<:Chisq})
    ηvec = getnaturalparameters(ef)
    @inbounds η1 = ηvec[1]
    
    return η1
end

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Chisq) =
    KnownExponentialFamilyDistribution(Chisq, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = unpack_naturalparameters(exponentialfamily)
    return Chisq(Int64(2 * (η + one(η))))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = unpack_naturalparameters(exponentialfamily)
    o = one(η)
    
    return loggamma(η + o) + (η + o) * log2
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = unpack_naturalparameters(exponentialfamily)
    return (η > minushalf)
end

struct OpenChi end
struct ClosedChi end

support(ef::KnownExponentialFamilyDistribution{Chisq}) = support(ef,check_boundaries(ef))
support(::KnownExponentialFamilyDistribution{Chisq},::OpenChi) = OpenInterval{Real}(0, Inf)
support(::KnownExponentialFamilyDistribution{Chisq},::ClosedChi) = ClosedInterval{Real}(0, Inf)
check_boundaries(ef::KnownExponentialFamilyDistribution{Chisq}) = unpack_naturalparameters(ef) == minushalf ? OpenChi() : ClosedChi()

support(dist::Chisq) = support(dist,check_boundaries(dist))
support(::Chisq,::OpenChi) = OpenInterval{Real}(0, Inf)
support(::Chisq,::ClosedChi) = ClosedInterval{Real}(0, Inf)
check_boundaries(dist::Chisq) = dof(dist) == 1 ? OpenChi() : ClosedChi()

function basemeasure(ef::KnownExponentialFamilyDistribution{Chisq}, x::Real)
    @assert insupport(ef, x) "$(x) is not in the support"
    return exp(-x / 2)
end
function fisherinformation(exponentialfamily::KnownExponentialFamilyDistribution{Chisq})
    η = unpack_naturalparameters(exponentialfamily)
    return [trigamma(η + one(η))]
end

function fisherinformation(dist::Chisq)
    return [trigamma(dof(dist) / 2) / 4]
end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Chisq}, x::Real)
    @assert insupport(ef, x) "$(x) is not in the support"
    return log(x)
end
