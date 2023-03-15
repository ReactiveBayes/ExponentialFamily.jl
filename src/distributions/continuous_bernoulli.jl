export ContinuousBernoulli
import StatsFuns: logexpm1, logistic
struct ContinuousBernoulli{T} <: ContinuousUnivariateDistribution
    λ::T
    ContinuousBernoulli(λ::T) where {T <: Real} = begin
        @assert 0 < λ < 1 "$(λ) needs to be in between 0 and 1"
        return new{T}(λ)
    end
end

vague(::Type{<:ContinuousBernoulli}) = ContinuousBernoulli(0.5)
succprob(dist::ContinuousBernoulli) = dist.λ
failprob(dist::ContinuousBernoulli) = 1 - succprob(dist)
probvec(dist::ContinuousBernoulli) = (failprob(dist), succprob(dist))

function Distributions.mean(dist::ContinuousBernoulli)
    λ = succprob(dist)
    if λ ≈ 1 / 2
        return 1 / 2
    else
        return λ / (2λ - 1) + 1 / (2atanh(1 - 2λ))
    end
end

function Distributions.var(dist::ContinuousBernoulli)
    λ = succprob(dist)
    if λ ≈ 1 / 2
        return 1 / 2
    else
        return λ * (1 - λ) / (1 - 2λ)^2 + 1 / (2atanh(1 - 2λ))^2
    end
end

function Distributions.cdf(dist::ContinuousBernoulli, x)
    λ = succprob(dist)
    if λ ≈ 1 / 2
        return x
    else
        return (λ^x * (1 - λ)^(1 - x) + λ - 1) / (2λ - 1)
    end
end

function Distributions.logpdf(dist::ContinuousBernoulli, x)
    @assert 0 <= x <= 1 "logpdf should be evaluated at a point between 0 and 1."
    λ = succprob(dist)

    if λ ≈ 1 / 2
        c = 2
    else
        c = 2atanh(1 - 2λ) / (1 - 2λ)
    end

    return c * x * log(λ) + (1 - x) * log(1 - λ)
end

Distributions.pdf(dist::ContinuousBernoulli, x) = exp(logpdf(dist, x))

prod_analytical_rule(::Type{<:ContinuousBernoulli}, ::Type{<:ContinuousBernoulli}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::ContinuousBernoulli, right::ContinuousBernoulli)
    npleft = convert(NaturalParameters, left)
    npright = convert(NaturalParameters, right)

    return convert(Distribution, npleft + npright)
end

function Base.convert(::Type{Distribution}, params::NaturalParameters{ContinuousBernoulli})
    logprobability = getindex(get_params(params), 1)
    return ContinuousBernoulli(logistic(logprobability))
end

function Base.convert(::Type{NaturalParameters}, dist::ContinuousBernoulli)
    @assert !(succprob(dist) ≈ 1) "Bernoulli natural parameters are not defiend for p = 1."
    NaturalParameters(ContinuousBernoulli, [log(succprob(dist) / (1 - succprob(dist)))])
end

isproper(params::NaturalParameters{ContinuousBernoulli}) = true

check_valid_natural(::Type{<:ContinuousBernoulli}, params) = (length(params) === 1)

basemeasure(T::Union{<:NaturalParameters{ContinuousBernoulli}, <:ContinuousBernoulli}, x) = 1.0

plus(::NaturalParameters{ContinuousBernoulli}, ::NaturalParameters{ContinuousBernoulli}) = Plus()

function lognormalizer(params::NaturalParameters{ContinuousBernoulli})
    η = first(get_params(params))
    if η ≈ 0.0
        return log(2)
    else
        return log((exp(η) - 1) / η + tiny)
    end
end
