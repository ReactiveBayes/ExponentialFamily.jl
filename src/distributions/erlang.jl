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

closed_prod_rule(::Type{<:Erlang}, ::Type{<:Erlang}) = ClosedProd()

function Base.prod(::ClosedProd, left::Erlang, right::Erlang)
    return Erlang(shape(left) + shape(right) - 1, (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

check_valid_natural(::Type{<:Erlang}, params) = length(params) === 2

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Erlang) =
    KnownExponentialFamilyDistribution(Erlang, [(shape(dist) - 1), -rate(dist)])

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Erlang})
    η = getnaturalparameters(exponentialfamily)
    a = first(η)
    b = getindex(η, 2)
    return Erlang(Int64(a + one(a)), -inv(b))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Erlang})
    η = getnaturalparameters(exponentialfamily)
    a = Int64(first(η))
    b = getindex(η, 2)
    return logfactorial(a) - (a + one(a)) * log(-b)
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Erlang})
    η = getnaturalparameters(exponentialfamily)
    a = first(η)
    b = getindex(η, 2)
    return (a >= tiny - 1) && (-b >= tiny)
end

support(::KnownExponentialFamilyDistribution{Erlang}) = ClosedInterval{Real}(0, Inf)

function basemeasure(ef::KnownExponentialFamilyDistribution{Erlang}, x::Real)
    @assert insupport(ef, x) "Erlang base measure should be evaluated at positive values"
    return one(x)
end
function fisherinformation(ef::KnownExponentialFamilyDistribution)
    η = getnaturalparameters(ef)
    η1 = first(η)
    η2 = getindex(η, 2)
    miη2 =-inv(η2)

    return SA[trigamma(η1) miη2; miη2 (η1+1)/(η2^2)]
end

function fisherinformation(dist::Erlang)
    k = shape(dist)
    λ = rate(dist)

    return SA[trigamma(k - 1) -inv(λ); -inv(λ) k/λ^2]
end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Erlang}, x::Real)
    @assert insupport(ef, x) "Erlang sufficientstatistics should be evaluated at positive values"
    return SA[log(x), x]
end
