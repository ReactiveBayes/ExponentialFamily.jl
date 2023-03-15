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

struct VagueContinuousBernoulli end
struct NonVagueContinuousBernoulli end

Distributions.mean(dist::ContinuousBernoulli) = mean(isvague(dist), dist)
Distributions.var(dist::ContinuousBernoulli) = mean(isvague(dist), dist)
Distributions.logpdf(dist::ContinuousBernoulli, x) = logpdf(isvague(dist), dist, x)
Distributions.pdf(dist::ContinuousBernoulli, x) = exp(Distributions.logpdf(dist, x))
Distributions.cdf(dist::ContinuousBernoulli, x) = cdf(isvague(dist), dist, x)

function isvague(dist::ContinuousBernoulli)
    if succprob(dist) ≈ 0.5
        return VagueContinuousBernoulli()
    else
        return NonVagueContinuousBernoulli()
    end
end
function mean(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli)
    λ = succprob(dist)
    return λ / (2λ - 1) + 1 / (2atanh(1 - 2λ))
end
mean(::VagueContinuousBernoulli, dist::ContinuousBernoulli) = 1 / 2

function var(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli)
    λ = succprob(dist)
    return λ * (1 - λ) / (1 - 2λ)^2 + 1 / (2atanh(1 - 2λ))^2
end
var(::VagueContinuousBernoulli, dist) = 1 / 2

function cdf(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli, x)
    @assert 0 <= x <= 1 "cdf should be evaluated at a point between 0 and 1."
    λ = succprob(dist)
    return (λ^x * (1 - λ)^(1 - x) + λ - 1) / (2λ - 1)
end
function cdf(::VagueContinuousBernoulli, dist::ContinuousBernoulli, x)
    @assert 0 <= x <= 1 "cdf should be evaluated at a point between 0 and 1."
    return x
end

function logpdf(::NonVagueContinuousBernoulli, dist::ContinuousBernoulli, x)
    @assert 0 <= x <= 1 "logpdf should be evaluated at a point between 0 and 1."
    λ = succprob(dist)
    c = 2atanh(1 - 2λ) / (1 - 2λ)
    return c * x * log(λ) + (1 - x) * log(1 - λ)
end

function logpdf(::VagueContinuousBernoulli, dist::ContinuousBernoulli, x)
    @assert 0 <= x <= 1 "logpdf should be evaluated at a point between 0 and 1."
    λ = succprob(dist)
    c = 2
    return c * x * log(λ) + (1 - x) * log(1 - λ)
end

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

function isvague(np::NaturalParameters{ContinuousBernoulli})
    if first(get_params(np)) ≈ 0.0
        return VagueContinuousBernoulli()
    else
        return NonVagueContinuousBernoulli()
    end
end

function lognormalizer(::NonVagueContinuousBernoulli,params::NaturalParameters{ContinuousBernoulli})
    η = first(get_params(params))
    return log((exp(η) - 1) / η + tiny)  
end
lognormalizer(::VagueContinuousBernoulli, params::NaturalParameters{ContinuousBernoulli}) = log(2.0)
lognormalizer(params::NaturalParameters{ContinuousBernoulli}) = lognormalizer(isvague(params),params)