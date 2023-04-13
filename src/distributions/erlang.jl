export Erlang

import SpecialFunctions: logfactorial, digamma
import Distributions: Erlang, shape, scale, cov

Distributions.cov(dist::Erlang) = var(dist)

function mean(::typeof(log), dist::Erlang)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

vague(::Type{<:Erlang}) = Erlang(1, huge)

prod_closed_rule(::Type{<:Erlang}, ::Type{<:Erlang}) = ClosedProd()

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
    a = first(η)
    b = getindex(η, 2)
    return logfactorial(a) - (a + one(a)) * log(-b)
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Erlang})
    η = getnaturalparameters(exponentialfamily)
    a = first(η)
    b = getindex(η, 2)
    return (a >= tiny - 1) && (-b >= tiny)
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Erlang}, <:Erlang}, x) = 1.0
