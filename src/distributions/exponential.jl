export Exponential

import Distributions: Exponential, params
import SpecialFunctions: digamma, logbeta

vague(::Type{<:Exponential}) = Exponential(Float64(huge))

prod_closed_rule(::Type{<:Exponential}, ::Type{<:Exponential}) = ClosedProd()

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

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Exponential)
    return KnownExponentialFamilyDistribution(Exponential, -inv(dist.θ))
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Exponential})
    return Exponential(-inv(getnaturalparameters(exponentialfamily)))
end

function logpartition(η::KnownExponentialFamilyDistribution{Exponential})
    return -log(-getnaturalparameters(η))
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{Exponential}) =
    (getnaturalparameters(exponentialfamily) <= zero(Float64))

support(::Union{<:KnownExponentialFamilyDistribution{Exponential}, <:Exponential}) = ClosedInterval{Real}(0, Inf)
insupport(union::Union{<:KnownExponentialFamilyDistribution{Exponential}, <:Exponential},x) = x ∈ support(union)

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Exponential}, <:Exponential}, x) 
    @assert insupport(union,x) "base measure should be evaluated at a point greater than 0"
    return one(typeof(x))
end
fisherinformation(exponentialfamily::KnownExponentialFamilyDistribution{Exponential}) =
    inv(getnaturalparameters(exponentialfamily)^2)

fisherinformation(dist::Exponential) = inv(dist.θ^2)
function sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{Exponential}, <:Exponential}, x) 
    @assert insupport(union,x) "sufficient statistics should be evaluated at a point greater than 0"
    return x
end