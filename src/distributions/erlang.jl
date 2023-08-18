export Erlang

import SpecialFunctions: logfactorial, digamma
import Distributions: Erlang, shape, scale, cov
using StaticArrays

Distributions.cov(dist::Erlang) = var(dist)

function mean(::typeof(log), dist::Erlang)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

vague(::Type{<:Erlang}) = Erlang(1, huge)

default_prod_rule(::Type{<:Erlang}, ::Type{<:Erlang}) = ClosedProd()

function Base.prod(::ClosedProd, left::Erlang, right::Erlang)
    return Erlang(shape(left) + shape(right) - 1, (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

# Natural parameterization

isproper(::NaturalParametersSpace, ::Type{Erlang}, η, conditioner) = isnothing(conditioner) && length(η) === 2 && isinteger(η[1]) && (η[1] >= 0) && (-η[2] >= tiny) && all(!isinf, η)
isproper(::MeanParametersSpace, ::Type{Erlang}, θ, conditioner) = isnothing(conditioner) && length(θ) === 2 && all(>(0), θ) && isinteger(θ[1]) && all(!isinf, θ)

function (::MeanToNatural{Erlang})(tuple_of_θ::Tuple{Any,Any})
    (shape, scale ) = tuple_of_θ
    return (shape - one(shape), -inv(scale) )
end

function (::NaturalToMean{Erlang})(tuple_of_η::Tuple{Any, Any})
    (η1,η2) = tuple_of_η
    return (η1 + one(η1), -inv(η2))
end

unpack_parameters(::Type{Erlang}, packed) = (first(packed), getindex(packed,2))

isbasemeasureconstant(::Type{Erlang}) = ConstantBaseMeasure()

getbasemeasure(::Type{Erlang}) = (x) -> one(x)
getsufficientstatistics(::Type{Erlang}) = (log, identity)

getlogpartition(::NaturalParametersSpace, ::Type{Erlang}) = (η) -> begin
    (η1, η2) = unpack_parameters(Erlang, η)

    return loggamma(η1+1) - (η1 + one(η1)) * log(-η2)
end

getfisherinformation(::NaturalParametersSpace, ::Type{Erlang}) = (η) -> begin
    (η1, η2)= unpack_parameters(Erlang, η)
    miη2 = -inv(η2)

    return SA[trigamma(η1+1) miη2; miη2 (η1+1)/(η2^2)]
end

# Mean parameterization

getlogpartition(::MeanParametersSpace, ::Type{Erlang}) = (θ) -> begin
    (k, β) = unpack_parameters(Erlang, θ)
    return k*log(β) + loggamma(k)
end

getfisherinformation(::MeanParametersSpace, ::Type{Erlang}) = (θ) -> begin
    (k, β ) = unpack_parameters(Erlang, θ)
    return SA[trigamma(k) -β; -β k*β^2]
end

























# check_valid_natural(::Type{<:Erlang}, params) = length(params) === 2

# pack_naturalparameters(dist::Erlang) = [(shape(dist) - 1), -rate(dist)]
# function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Erlang})
#     η = getnaturalparameters(ef)
#     @inbounds η1 = η[1]
#     @inbounds η2 = η[2]

#     return η1, η2
# end

# Base.convert(::Type{ExponentialFamilyDistribution}, dist::Erlang) =
#     ExponentialFamilyDistribution(Erlang, pack_naturalparameters(dist))

# function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Erlang})
#     a, b = unpack_naturalparameters(exponentialfamily)
#     return Erlang(Int64(a + one(a)), -inv(b))
# end

# function logpartition(exponentialfamily::ExponentialFamilyDistribution{Erlang})
#     a, b = unpack_naturalparameters(exponentialfamily)
#     inta = Int64(a)
#     return logfactorial(inta) - (inta + one(inta)) * log(-b)
# end

# function isproper(exponentialfamily::ExponentialFamilyDistribution{Erlang})
#     a, b = unpack_naturalparameters(exponentialfamily)
#     return (a >= tiny - 1) && (-b >= tiny)
# end

# support(::ExponentialFamilyDistribution{Erlang}) = ClosedInterval{Real}(0, Inf)

# basemeasure(::ExponentialFamilyDistribution{Erlang}) = one(Float64)
# basemeasure(::ExponentialFamilyDistribution{Erlang}, x::Real) = one(x)

# function fisherinformation(ef::ExponentialFamilyDistribution)
#     η1, η2 = unpack_naturalparameters(ef)
#     miη2 = -inv(η2)

#     return SA[trigamma(η1) miη2; miη2 (η1+1)/(η2^2)]
# end

# function fisherinformation(dist::Erlang)
#     k = shape(dist)
#     λ = rate(dist)

#     return SA[trigamma(k - 1) -inv(λ); -inv(λ) k/λ^2]
# end

# sufficientstatistics(ef::ExponentialFamilyDistribution{Erlang}) = (x) -> sufficientstatistics(ef, x)
# sufficientstatistics(::ExponentialFamilyDistribution{Erlang}, x::Real) = SA[log(x), x]
