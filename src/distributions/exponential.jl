export Exponential

import Distributions: Exponential, params
import SpecialFunctions: digamma, logbeta
using StaticArrays

vague(::Type{<:Exponential}) = Exponential(Float64(huge))

closed_prod_rule(::Type{<:Exponential}, ::Type{<:Exponential}) = ClosedProd()

function Base.prod(::ClosedProd, left::Exponential, right::Exponential)
    invθ_left  = inv(left.θ)
    invθ_right = inv(right.θ)
    return Exponential(inv(invθ_left + invθ_right))
end

function mean(::typeof(log), dist::Exponential)
    return -log(rate(dist)) - MathConstants.eulergamma
end

function logpartition(dist::Exponential)
    return -log(rate(dist))
end

check_valid_natural(::Type{<:Exponential}, params) = length(params) === 1

pack_naturalparameters(dist::Exponential) = [-inv(dist.θ)]
unpack_naturalparameters(ef::ExponentialFamilyDistribution) = (first(getnaturalparameters(ef)),)
function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Exponential)
    return ExponentialFamilyDistribution(Exponential, pack_naturalparameters(dist))
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Exponential})
    return Exponential(-inv(first(unpack_naturalparameters(exponentialfamily))))
end

function logpartition(ef::ExponentialFamilyDistribution{Exponential})
    return -log(-first(unpack_naturalparameters(ef)))
end

isproper(exponentialfamily::ExponentialFamilyDistribution{Exponential}) =
    (first(unpack_naturalparameters(exponentialfamily)) <= zero(Float64))

support(::Union{<:ExponentialFamilyDistribution{Exponential}, <:Exponential}) = ClosedInterval{Real}(0, Inf)

basemeasure(::ExponentialFamilyDistribution{Exponential}) = one(Float64)
basemeasure(::ExponentialFamilyDistribution{Exponential}, x::Real) = one(x)

fisherinformation(exponentialfamily::ExponentialFamilyDistribution{Exponential}) =
    SA[inv(first(unpack_naturalparameters(exponentialfamily))^2);;]

fisherinformation(dist::Exponential) = SA[inv(dist.θ^2);;]

sufficientstatistics(ef::ExponentialFamilyDistribution{Exponential}) = (x) -> sufficientstatistics(ef, x)
sufficientstatistics(::ExponentialFamilyDistribution{Exponential}, x::Real) = SA[x]
