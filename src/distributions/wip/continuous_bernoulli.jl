export ContinuousBernoulli
import StatsFuns: logexpm1, logistic
using Distributions
using StaticArrays

using Random
import LogExpFunctions: logexpm1

"""
    ContinuousBernoulli{T}

A univariate continuous Bernoulli distribution parametrized by its success probability `λ`.

# Fields
- `λ`: The success probability of the continuous Bernoulli distribution. It should be a real number in the interval (0, 1)
"""
struct ContinuousBernoulli{T} <: ContinuousUnivariateDistribution
    λ::T
    ContinuousBernoulli(λ::T) where {T <: Real} = begin
        @assert zero(λ) < λ < one(λ) "$(λ) needs to be in between 0 and 1"
        return new{T}(λ)
    end
end

vague(::Type{<:ContinuousBernoulli}) = ContinuousBernoulli(0.5)
succprob(dist::ContinuousBernoulli) = dist.λ
failprob(dist::ContinuousBernoulli) = one(Float64) - succprob(dist)
probvec(dist::ContinuousBernoulli) = (failprob(dist), succprob(dist))

struct VagueContinuousBernoulli end
struct NonVagueContinuousBernoulli end

function isvague(dist::ContinuousBernoulli)
    if succprob(dist) ≈ 0.5
        return VagueContinuousBernoulli()
    else
        return NonVagueContinuousBernoulli()
    end
end

Distributions.mean(dist::ContinuousBernoulli) = mean(isvague(dist), dist)
Distributions.var(dist::ContinuousBernoulli) = var(isvague(dist), dist)
function Distributions.logpdf(dist::ContinuousBernoulli, x::Real)
    @assert zero(x) <= x <= one(x) "Second argument to logpdf should be a probability in between 0 and 1"
    return logpdf(isvague(dist), dist, x)
end
function Distributions.pdf(dist::ContinuousBernoulli, x::Real)
    @assert zero(x) <= x <= one(x) "Second argument to logpdf should be a probability in between 0 and 1"
    return exp(Distributions.logpdf(dist, x))
end
Distributions.cdf(dist::ContinuousBernoulli, x::Real) = cdf(isvague(dist), dist, x)
function icdf(dist::ContinuousBernoulli, x)
    @assert zero(x) <= x <= one(x) "Second argument to icdf should be a probability in between 0 and 1"
    return icdf(isvague(dist), dist, x)
end

function mean(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli)
    λ = succprob(dist)
    return λ / (2 * λ - oneunit(λ)) + oneunit(λ) / (2 * atanh(oneunit(λ) - 2 * λ))
end
mean(::VagueContinuousBernoulli, dist::ContinuousBernoulli) = 1 / 2

function var(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli)
    (η,) = unpack_naturalparameters(convert(ExponentialFamilyDistribution, dist))
    eη = exp(η)
    return (-eη * (η^2 + 2) + eη^2 + 1) / ((eη - 1)^2 * η^2)
end

var(::VagueContinuousBernoulli, dist) = 1 / 12

function cdf(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli, x::Real)
    @assert zero(x) <= x <= one(x) "cdf should be evaluated at a point between 0 and 1."
    λ = succprob(dist)
    return (λ^x * (1 - λ)^(1 - x) + λ - 1) / (2λ - 1)
end
function cdf(::VagueContinuousBernoulli, dist::ContinuousBernoulli, x::Real)
    @assert zero(x) <= x <= one(x) "cdf should be evaluated at a point between 0 and 1."
    return x
end

icdf(::VagueContinuousBernoulli, dist::ContinuousBernoulli, x::Real) = x
function icdf(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli, x)
    λ = succprob(dist)
    term1 = log((2λ - 1) * x + 1 - λ) - log(1 - λ)
    term2 = log(λ) - log(1 - λ)

    return term1 / term2
end

function logpdf(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli, x::Real)
    @assert 0 <= x <= 1 "logpdf should be evaluated at a point between 0 and 1."
    λ = succprob(dist)
    c = 2atanh(1 - 2λ) / (1 - 2λ)
    return x * log(λ) + (1 - x) * log(1 - λ) + log(c)
end

function logpdf(::VagueContinuousBernoulli, dist::ContinuousBernoulli, x::Real)
    @assert 0 <= x <= 1 "logpdf should be evaluated at a point between 0 and 1."
    λ = succprob(dist)
    c = 2.0
    return x * log(λ) + (1 - x) * log(1 - λ) + log(c)
end

closed_prod_rule(::Type{<:ContinuousBernoulli}, ::Type{<:ContinuousBernoulli}) = ClosedProd()

pack_naturalparameters(dist::ContinuousBernoulli) = [log(succprob(dist) / (one(Float64) - succprob(dist)))]
function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:ContinuousBernoulli})
    η = getnaturalparameters(ef)
    @inbounds logprobability = η[1]

    return (logprobability,)
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{ContinuousBernoulli})
    return ContinuousBernoulli(logistic(first(unpack_naturalparameters(exponentialfamily))))
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::ContinuousBernoulli)
    ExponentialFamilyDistribution(ContinuousBernoulli, pack_naturalparameters(dist))
end

isproper(exponentialfamily::ExponentialFamilyDistribution{ContinuousBernoulli}) = true

check_valid_natural(::Type{<:ContinuousBernoulli}, params) = (length(params) === 1)

function isvague(exponentialfamily::ExponentialFamilyDistribution{ContinuousBernoulli})
    if first(unpack_naturalparameters(exponentialfamily)) ≈ 0.0
        return VagueContinuousBernoulli()
    else
        return NonVagueContinuousBernoulli()
    end
end

function logpartition(
    ::NonVagueContinuousBernoulli,
    exponentialfamily::ExponentialFamilyDistribution{ContinuousBernoulli}
)
    (η,) = unpack_naturalparameters(exponentialfamily)
    return log((exp(η) - 1) / η + tiny)
end
logpartition(::VagueContinuousBernoulli, exponentialfamily::ExponentialFamilyDistribution{ContinuousBernoulli}) =
    log(2.0)
logpartition(exponentialfamily::ExponentialFamilyDistribution{ContinuousBernoulli}) =
    logpartition(isvague(exponentialfamily), exponentialfamily)

Random.rand(rng::AbstractRNG, dist::ContinuousBernoulli{T}) where {T} = icdf(dist, rand(rng, Uniform()))

function Random.rand(rng::AbstractRNG, dist::ContinuousBernoulli{T}, size::Int64) where {T}
    container = Array{T}(undef, size)
    return rand!(rng, dist, container)
end

function Random.rand!(rng::AbstractRNG, dist::ContinuousBernoulli, container::AbstractArray{T}) where {T <: Real}
    @inbounds for i in 1:size(container, 1)
        temp = rand(rng, dist)
        @views container[i] = temp
    end
    return container
end

fisherinformation(ef::ExponentialFamilyDistribution{ContinuousBernoulli}) = fisherinformation(isvague(ef), ef)
fisherinformation(::VagueContinuousBernoulli, ef::ExponentialFamilyDistribution{ContinuousBernoulli}) = SA[1 / 12;;]
function fisherinformation(::NonVagueContinuousBernoulli, ef::ExponentialFamilyDistribution{ContinuousBernoulli})
    (η,) = unpack_naturalparameters(ef)
    return SA[inv(η^2) - exp(η) / (exp(η) - 1)^2;;]
end

fisherinformation(dist::ContinuousBernoulli) = fisherinformation(isvague(dist), dist)
fisherinformation(::VagueContinuousBernoulli, dist::ContinuousBernoulli) = SA[16 / 12;;]
function fisherinformation(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli)
    λ = succprob(dist)
    m = mean(dist)
    tmp1 = (2 - 4λ) * atanh(1 - 2λ) - 1
    tmp2 = 4 * (λ - 1)^2 * λ^2 * (atanh(1 - 2λ)^2)
    return SA[m / λ^2 + (1 - m) / (1 - λ)^2 - 4 / (1 - 2λ)^2 - tmp1 / tmp2;;]
end

function support(::Union{<:ExponentialFamilyDistribution{ContinuousBernoulli}, <:ContinuousBernoulli})
    return ClosedInterval{Real}(0.0, 1.0)
end

function insupport(ef::ExponentialFamilyDistribution{ContinuousBernoulli, P, C, Safe}, x::Real) where {P, C}
    return x ∈ support(ef)
end

function insupport(dist::ContinuousBernoulli, x::Real)
    return x ∈ support(dist)
end

sufficientstatistics(ef::ExponentialFamilyDistribution{<:ContinuousBernoulli}) = (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(
    ::ExponentialFamilyDistribution{ContinuousBernoulli},
    x::Real
)
    return SA[x]
end

basemeasure(ef::ExponentialFamilyDistribution{<:ContinuousBernoulli}) = basemeasure(ef, isvague(ef))
basemeasure(ef::ExponentialFamilyDistribution{<:ContinuousBernoulli}, ::VagueContinuousBernoulli) =
    exp(logpartition(ef))
basemeasure(::ExponentialFamilyDistribution{<:ContinuousBernoulli}, ::NonVagueContinuousBernoulli) = one(Float64)
basemeasure(
    exponentialfamily::ExponentialFamilyDistribution{<:ContinuousBernoulli},
    x::Real
) =
    basemeasure(isvague(exponentialfamily), exponentialfamily, x)

function basemeasure(
    ::VagueContinuousBernoulli,
    ef::ExponentialFamilyDistribution{<:ContinuousBernoulli},
    x::Real
)
    return exp(logpartition(ef))
end

function basemeasure(
    ::NonVagueContinuousBernoulli,
    ef::ExponentialFamilyDistribution{ContinuousBernoulli},
    x::Real
)
    return one(x)
end
