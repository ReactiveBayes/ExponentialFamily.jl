export NormalGamma
using Distributions
import StatsFuns: loggamma
using Random

struct NormalGamma{T <: Real} <: ContinuousMultivariateDistribution
    μ::T
    λ::T
    α::T
    β::T
end

function NormalGamma(μ::Real, λ::Real, α::Real, β::Real)
    μ_promoted, λ_promoted, α_promoted, β_promoted = promote(μ, λ, α, β)
    T = typeof(μ_promoted)
    return NormalGamma{T}(μ_promoted, λ_promoted, α_promoted, β_promoted)
end

params(d::NormalGamma)   = (d.μ, d.λ, d.α, d.β)
location(d::NormalGamma) = first(params(d))
scale(d::NormalGamma)    = getindex(params(d), 2)
shape(d::NormalGamma)    = getindex(params(d), 3)
rate(d::NormalGamma)     = getindex(params(d), 4)

prod_closed_rule(::Type{<:NormalGamma}, ::Type{<:NormalGamma}) = ClosedProd()

check_valid_natural(::Type{<:NormalGamma}, params) = length(params) === 4

function Distributions.pdf(dist::NormalGamma, x::AbstractVector{<:Real})
    ef = convert(KnownExponentialFamilyDistribution, dist)
    η  = getnaturalparameters(ef)
    Tx = sufficientstatistics(ef)(x...)
    return basemeasure(dist, x) * exp(η'Tx - logpartition(ef))
end

Distributions.logpdf(dist::NormalGamma, x::AbstractVector{<:Real}) = log(pdf(dist, x))

sufficientstatistics(::Union{<:KnownExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}) =
    (x, τ) -> [τ * x, τ * x^2, log(τ), τ]

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::NormalGamma)
    μ, λ, α, β = params(dist)
    η1 = λ * μ
    η2 = -λ / 2
    η3 = α - 1 / 2
    η4 = -β - λ * μ^2 / 2
    η = [η1, η2, η3, η4]

    return KnownExponentialFamilyDistribution(NormalGamma, η)
end
function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = getnaturalparameters(exponentialfamily)
    return NormalGamma(-η1 / 2η2, -2η2, η3 + 1 / 2, -η4 + η3^2 / 4η4)
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = getnaturalparameters(exponentialfamily)
    return loggamma(η3 + 1 / 2) - log(-2η2) / 2 - (η3 + 1 / 2) * log(-η4 + η1^2 / (4η2))
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = getnaturalparameters(exponentialfamily)
    return -η2 > 0 && (η3 >= tiny - 1 / 2) && (-η4 >= tiny)
end

basemeasure(d::Union{<:KnownExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}, x) = 1 / sqrt(2π)

function Random.rand!(rng::AbstractRNG, dist::NormalGamma, container::Tuple)
    rand!(rng, GammaShapeRate(dist.α, dist.β), container[2])
    rand!(rng, NormalMeanPrecision(dist.μ, dist.λ * first(container[2])), first(container))
    return container
end

function Random.rand!(rng::AbstractRNG, dist::NormalGamma, container::AbstractVector{T}) where {T <: Tuple}
    for i in eachindex(container)
        rand!(rng, dist, container[i])
    end
    return container
end

function Random.rand(rng::AbstractRNG, dist::NormalGamma)
    container = (Vector{Float64}(undef, 1), Vector{Float64}(undef, 1))
    rand!(rng, dist, container)
    return container
end

function Random.rand(rng::AbstractRNG, dist::NormalGamma, nsamples::Int)
    container = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, nsamples)
    for i in eachindex(container)
        container[i] = (Vector{Float64}(undef, 1), Vector{Float64}(undef, 1))
        rand!(rng, dist, container[i])
    end
    return container
end
