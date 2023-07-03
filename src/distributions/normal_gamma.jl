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

mean(d::NormalGamma) = [d.μ, d.α / d.β]
var(d::NormalGamma) = [d.β / (d.λ * (d.α - 1)), d.α / (d.β^2)]
cov(d::NormalGamma) = [d.β/(d.λ*(d.α-1)) 0.0; 0.0 d.α/(d.β^2)]

closed_prod_rule(::Type{<:NormalGamma}, ::Type{<:NormalGamma}) = ClosedProd()

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

sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}, x) =
    sufficientstatistics(union)(x...)

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::NormalGamma)
    μ, λ, α, β = params(dist)
    η1 = λ * μ
    η2 = -λ / 2
    η3 = α - (1 / 2)
    η4 = -β - λ * μ^2 / 2
    η = [η1, η2, η3, η4]

    return KnownExponentialFamilyDistribution(NormalGamma, η)
end
function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = getnaturalparameters(exponentialfamily)
    return NormalGamma(-η1 / (2η2), -2η2, η3 + (1 / 2), -η4 + (η1^2 / 4η2))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = getnaturalparameters(exponentialfamily)
    return loggamma(η3 + 1 / 2) - log(-2η2) / 2 - (η3 + 1 / 2) * log(-η4 + η1^2 / (4η2))
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{NormalGamma})
    _, η2, η3, η4 = getnaturalparameters(exponentialfamily)
    return -η2 > 0 && (η3 >= tiny - 1 / 2) && (-η4 >= tiny)
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}, x) = 1 / sqrt(2π)

function Random.rand!(rng::AbstractRNG, dist::NormalGamma, container::AbstractVector)
    container[2] = rand(rng, GammaShapeRate(dist.α, dist.β))
    container[1] = rand(rng, NormalMeanPrecision(dist.μ, dist.λ * container[2]))
    return container
end

function Random.rand!(rng::AbstractRNG, dist::NormalGamma, container::AbstractVector{T}) where {T <: Vector}
    for i in eachindex(container)
        rand!(rng, dist, container[i])
    end
    return container
end

function Random.rand(rng::AbstractRNG, dist::NormalGamma)
    container = Vector{Real}(undef, 2)
    rand!(rng, dist, container)
    return container
end

function Random.rand(rng::AbstractRNG, dist::NormalGamma, nsamples::Int)
    container = Vector{Vector{Real}}(undef, nsamples)
    for i in eachindex(container)
        container[i] = Vector{Real}(undef, 2)
        rand!(rng, dist, container[i])
    end
    return container
end

function fisherinformation(exponentialfamily::KnownExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = getnaturalparameters(exponentialfamily)

    # Define a 4x4 matrix
    info_matrix = Matrix{Float64}(undef, 4, 4)

    # Assign each value to the corresponding cell in the matrix
    info_matrix[1, 1] =
        ((((η1^2) / (4η2) - η4)^-1) * (-0.5 - η3) + (-(η1^2) * (((η1^2) / (4η2) - η4)^-2) * (-0.5 - η3)) / (2η2)) /
        (2η2)
    info_matrix[2, 1] =
        (-η1 * (((η1^2) / (4η2) - η4)^-1) * (-0.5 - η3)) / (2 * (η2^2)) +
        (2η1 * ((η1^2) / (16 * (η2^2))) * (((η1^2) / (4η2) - η4)^-2) * (-0.5 - η3)) / η2
    info_matrix[3, 1] = (-η1 * (((η1^2) / (4η2) - η4)^-1)) / (2η2)
    info_matrix[4, 1] = (η1 * (((η1^2) / (4η2) - η4)^-2) * (-0.5 - η3)) / (2η2)
    info_matrix[1, 2] =
        ((η1^3) * (((η1^2) / (4η2) - η4)^-2) * (-0.5 - η3)) / (8 * (η2^3)) +
        (-η1 * (((η1^2) / (4η2) - η4)^-1) * (-0.5 - η3)) / (2 * (η2^2))
    info_matrix[2, 2] =
        (1 // 2) * (η2^-2) + (-(η1^2) * ((η1^2) / (16 * (η2^2))) * (((η1^2) / (4η2) - η4)^-2) * (-0.5 - η3)) / (η2^2) +
        128η2 * ((η1^2) / (256 * (η2^4))) * (((η1^2) / (4η2) - η4)^-1) * (-0.5 - η3)
    info_matrix[3, 2] = ((η1^2) * (((η1^2) / (4η2) - η4)^-1)) / (4 * (η2^2))
    info_matrix[4, 2] = (-(η1^2) * (((η1^2) / (4η2) - η4)^-2) * (-0.5 - η3)) / (4 * (η2^2))
    info_matrix[1, 3] = (-η1 * (((η1^2) / (4η2) - η4)^-1)) / (2η2)
    info_matrix[2, 3] = 4 * ((η1^2) / (16 * (η2^2))) * (((η1^2) / (4η2) - η4)^-1)
    info_matrix[3, 3] = SpecialFunctions.trigamma(0.5 + η3)
    info_matrix[4, 3] = ((η1^2) / (4η2) - η4)^-1
    info_matrix[1, 4] = (-η1 * (0.5 + η3) * (((η1^2) / (4η2) - η4)^-2)) / (2η2)
    info_matrix[2, 4] = 4 * (0.5 + η3) * ((η1^2) / (16 * (η2^2))) * (((η1^2) / (4η2) - η4)^-2)
    info_matrix[3, 4] = ((η1^2) / (4η2) - η4)^-1
    info_matrix[4, 4] = (0.5 + η3) * (((η1^2) / (4η2) - η4)^-2)

    return info_matrix
end

function fisherinformation(dist::NormalGamma)
    μ, λ, α, β = params(dist)

    # Define a 4x4 matrix
    info_matrix = Matrix{Float64}(undef, 4, 4)

    # Assign each value to the corresponding cell in the matrix
    info_matrix[1, 1] = -λ * α / β
    info_matrix[2, 1] = 0
    info_matrix[3, 1] = 0
    info_matrix[4, 1] = 0
    info_matrix[1, 2] = 0
    info_matrix[2, 2] = -0.5 * (λ^-2)
    info_matrix[3, 2] = 0
    info_matrix[4, 2] = 0
    info_matrix[1, 3] = 0
    info_matrix[2, 3] = 0
    info_matrix[3, 3] = -SpecialFunctions.trigamma(α)
    info_matrix[4, 3] = β^-1
    info_matrix[1, 4] = 0
    info_matrix[2, 4] = 0
    info_matrix[3, 4] = β^-1
    info_matrix[4, 4] = -α * (β^-2)

    # Return the resulting information matrix
    return -info_matrix
end
