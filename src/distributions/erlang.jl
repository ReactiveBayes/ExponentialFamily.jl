export Erlang, ErlangNaturalParameters

import SpecialFunctions: logfactorial, digamma
import Distributions: Erlang, shape, scale, cov

Distributions.cov(dist::Erlang) = var(dist)

function mean(::typeof(log), dist::Erlang)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

vague(::Type{<:Erlang}) = Erlang(1, huge)

# Extensions of prod methods

prod_analytical_rule(::Type{<:Erlang}, ::Type{<:Erlang}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Erlang, right::Erlang)
    return Erlang(shape(left) + shape(right) - 1, (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

## Friendly functions

function logpdf_sample_friendly(dist::Erlang)
    k, λ = params(dist)
    friendly = Erlang(k, λ)
    return (friendly, friendly)
end

## Natural parameters for the Erlang distribution

struct ErlangNaturalParameters <: NaturalParameters
    a::Integer
    b::Real
end

ErlangNaturalParameters(a::Real, b::Real)       = ErlangNaturalParameters(Int(a), b)
ErlangNaturalParameters(a::Integer, b::Integer) = ErlangNaturalParameters(a, float(b))

function ErlangNaturalParameters(vec::AbstractVector)
    @assert length(vec) === 2 "`ErlangNaturalParameters` must accept a vector of length `2`."
    return ErlangNaturalParameters(vec[1], vec[2])
end

as_naturalparams(::Type{T}, args...) where {T <: ErlangNaturalParameters} = convert(ErlangNaturalParameters, args...)
naturalparams(dist::Erlang) = ErlangNaturalParameters(shape(dist) - 1, -rate(dist))

Base.convert(::Type{ErlangNaturalParameters}, a::Real, b::Real) = ErlangNaturalParameters(convert(Int, a), b)

function Base.:(==)(left::ErlangNaturalParameters, right::ErlangNaturalParameters)
    return left.a == right.a && left.b == right.b
end

function Base.convert(::Type{Distribution}, params::ErlangNaturalParameters)
    return Erlang(params.a + 1, -1 / params.b)
end

function Base.vec(params::ErlangNaturalParameters)
    return [params.a, params.b]
end

function Base.:+(left::ErlangNaturalParameters, right::ErlangNaturalParameters)
    return ErlangNaturalParameters(left.a + right.a, left.b + right.b)
end

function Base.:-(left::ErlangNaturalParameters, right::ErlangNaturalParameters)
    return ErlangNaturalParameters(left.a - right.a, left.b - right.b)
end

function lognormalizer(params::ErlangNaturalParameters)
    return logfactorial(params.a) - (params.a + 1) * log(-params.b)
end

function Distributions.logpdf(params::ErlangNaturalParameters, x)
    return log(x) * params.a + x * params.b - lognormalizer(params)
end

function isproper(params::ErlangNaturalParameters)
    return (params.a >= tiny - 1) && (-params.b >= tiny)
end
