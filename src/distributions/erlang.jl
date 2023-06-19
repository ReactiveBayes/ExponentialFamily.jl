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

function insupport(union::Union{<:KnownExponentialFamilyDistribution{Erlang}, <:Erlang},x)
    return x ∈ support(union)
end

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Erlang}, <:Erlang}, x) 
    @assert insupport(union,x) "Erlang base measure should be evaluated at positive values"
    return one(typeof(x))
end
function fisherinformation(ef::KnownExponentialFamilyDistribution)
    η = getnaturalparameters(ef)
    η1 = first(η)
    η2 = getindex(η, 2)

    return [trigamma(η1) -inv(η2); -inv(η2) (η1+1)/(η2^2)]
end

function fisherinformation(dist::Erlang)
    k = shape(dist)
    λ = rate(dist)

    return [trigamma(k - 1) -inv(λ); -inv(λ) k/λ^2]
end

function sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{Erlang}, <:Erlang}, x) 
    @assert insupport(union,x) "Erlang sufficientstatistics should be evaluated at positive values"
    return [log(x), x]
end