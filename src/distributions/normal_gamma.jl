export NormalGamma
using Distributions
import StatsFuns: loggamma
using Random
using StaticArrays

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
    ef = convert(ExponentialFamilyDistribution, dist)
    η  = getnaturalparameters(ef)
    Tx = sufficientstatistics(ef)(x...)
    return basemeasure(dist, x) * exp(η'Tx - logpartition(ef))
end

Distributions.logpdf(dist::NormalGamma, x::AbstractVector{<:Real}) = log(pdf(dist, x))

sufficientstatistics(::Union{<:ExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}) =
    (x, τ) -> SA[τ * x, τ * x^2, log(τ), τ]

sufficientstatistics(union::Union{<:ExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}, x) =
    sufficientstatistics(union)(x[1], x[2])

function pack_naturalparameters(dist::NormalGamma) 
    μ, λ, α, β = params(dist)
    η1 = λ * μ
    η2 = -λ * HALF
    η3 = α - HALF
    η4 = -β - λ * μ^2 * HALF

    return [η1, η2, η3, η4]
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<: NormalGamma})
    η =  getnaturalparameters(ef)
    @inbounds η1 = η[1]
    @inbounds η2 = η[2]
    @inbounds η3 = η[3]
    @inbounds η4 = η[4]

    return η1, η2, η3, η4
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::NormalGamma) = ExponentialFamilyDistribution(NormalGamma, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)
    return NormalGamma(η1*MINUSHALF/ (η2), -2η2, η3 + HALF, -η4 + (η1^2 / 4η2))
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)
    η3half = η3 + HALF
    return loggamma(η3half) - log(-2η2) * HALF - (η3half) * log(-η4 + η1^2 / (4η2))
end

function isproper(exponentialfamily::ExponentialFamilyDistribution{NormalGamma})
    _, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)
    return -η2 > 0 && (η3 >= tiny + minushalf) && (-η4 >= tiny)
end

basemeasure(::Union{<:ExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}, x) = 1 / SQRT2PI

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

##fisher information should be optimized further
function fisherinformation(exponentialfamily::ExponentialFamilyDistribution{NormalGamma})
    η1, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)

    # Define a 4x4 matrix
    info_matrix = Matrix{Float64}(undef, 4, 4)
    tmp1 = ((η1^2) / (4η2) - η4)^-1
    tmp2 = ((η1^2) / (4η2) - η4)^-2
    # Assign each value to the corresponding cell in the matrix
    info_matrix[1, 1] =
        ((tmp1) * (MINUSHALF - η3) + (-(η1^2) * (tmp2) * (MINUSHALF - η3)) / (2η2)) /
        (2η2)
    info_matrix[2, 1] =
        (-η1 * (tmp1) * (MINUSHALF - η3)) / (2 * (η2^2)) +
        (2η1 * ((η1^2) / (16 * (η2^2))) * (tmp2) * (MINUSHALF - η3)) / η2
    info_matrix[3, 1] = (-η1 * (tmp1)) / (2η2)
    info_matrix[4, 1] = (η1 * (tmp2) * (MINUSHALF - η3)) / (2η2)
    info_matrix[1, 2] =
        ((η1^3) * (tmp2) * (MINUSHALF - η3)) / (8 * (η2^3)) +
        (-η1 * (tmp1) * (MINUSHALF - η3)) / (2 * (η2^2))
    info_matrix[2, 2] =
        (1 // 2) * (η2^-2) + (-(η1^2) * ((η1^2) / (16 * (η2^2))) * (tmp2) * (MINUSHALF - η3)) / (η2^2) +
        128η2 * ((η1^2) / (256 * (η2^4))) * (tmp1) * (MINUSHALF - η3)
    info_matrix[3, 2] = ((η1^2) * (tmp1)) / (4 * (η2^2))
    info_matrix[4, 2] = (-(η1^2) * (tmp2) * (MINUSHALF - η3)) / (4 * (η2^2))
    info_matrix[1, 3] = (-η1 * (tmp1)) / (2η2)
    info_matrix[2, 3] = 4 * ((η1^2) / (16 * (η2^2))) * (tmp1)
    info_matrix[3, 3] = SpecialFunctions.trigamma(HALF + η3)
    info_matrix[4, 3] = tmp1
    info_matrix[1, 4] = (-η1 * (HALF + η3) * (tmp2)) / (2η2)
    info_matrix[2, 4] = 4 * (HALF + η3) * ((η1^2) / (16 * (η2^2))) * (tmp2)
    info_matrix[3, 4] = tmp1
    info_matrix[4, 4] = (HALF + η3) * (tmp2)

    return info_matrix
end

## Allocates for zeros
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
    info_matrix[2, 2] = MINUSHALF * (λ^-2)
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
