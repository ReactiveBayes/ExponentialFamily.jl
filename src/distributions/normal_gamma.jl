export NormalGamma
using Distributions
import StatsFuns: loggamma
using Random
using StaticArrays
using DomainSets

"""
    NormalGamma{T <: Real} <: ContinuousMultivariateDistribution

A normal-gamma distribution, where `T` is a real number. This distribution is a joint distribution of a normal random variable with mean `μ` and precision `λ`, and a gamma-distributed random variable with shape `α` and rate `β`.

# Fields
- `μ::T`: The mean of the normal distribution.
- `λ::T`: The precision of the normal distribution.
- `α::T`: The shape parameter of the gamma distribution.
- `β::T`: The rate parameter of the gamma distribution.
"""
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
var(d::NormalGamma) = [d.β / (d.λ * (d.α - one(d.α))), d.α / (d.β^2)]
cov(d::NormalGamma) = [d.β/(d.λ*(d.α - one(d.α ))) 0.0; 0.0 d.α/(d.β^2)]
std(d::NormalGamma) = sqrt.(var(d))

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
    container = Vector{Float64}(undef, 2)
    rand!(rng, dist, container)
    return container
end

function Random.rand(rng::AbstractRNG, dist::NormalGamma, nsamples::Int)
    container = Vector{Vector{Float64}}(undef, nsamples)
    for i in eachindex(container)
        container[i] = Vector{Float64}(undef, 2)
        rand!(rng, dist, container[i])
    end
    return container
end

function Distributions.logpdf(dist::NormalGamma, x::AbstractVector{<:Real})
    (μ, λ, α, β) = params(dist)

    constants = α*log(β)+ (1/2)*(log(λ)) - loggamma(α) - (1/2)*log(twoπ)
    term1 = (α - 1/2)*log(x[2]) - β*x[2]
    term2 = -λ*x[2]*((x[1]-μ)^2)/2

    return constants + term1 + term2
end

Distributions.pdf(dist::NormalGamma, x::AbstractVector{<:Real}) = exp(logpdf(dist, x))

default_prod_rule(::Type{<:NormalGamma}, ::Type{<:NormalGamma}) = PreserveTypeProd(Distribution)

function Base.prod(::PreserveTypeProd{Distribution}, left::NormalGamma, right::NormalGamma)
    (μleft, λleft, αleft, βleft) = params(left)
    (μright, λright, αright, βright) = params(right)

    λ = λleft + λright
    μ = (λleft*μleft + λright*μright)/λ
    α = αleft + αright - 1/2
    ## β term could be problematic
    β = βleft + βright + λleft*μleft^2/2 + λright*μright^2/2 + (λleft*μleft + λright*μright)^2/(-2*λ)
    
    return NormalGamma(μ,λ,α,β)
end

struct NormalGammaDomain <: Domain{AbstractVector} end

Base.eltype(::NormalGammaDomain) = AbstractVector
Base.in(v, ::NormalGammaDomain) = length(v) === 2 && isreal(v[1]) && isreal(v[2]) && v[2] > 0

Distributions.support(::Type{NormalGamma}) = NormalGammaDomain()

# Natural parametrization
isproper(::NaturalParametersSpace, ::Type{NormalGamma}, η, conditioner) = isnothing(conditioner) && length(η) === 4 && getindex(η,2) < 0 && getindex(η, 3) > -1/2 && getindex(η,4) < 0 && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{NormalGamma}, θ, conditioner) = isnothing(conditioner) && length(θ) === 4 && all(>(0), getindex(θ,2:4)) && all(!isinf, θ) && all(!isnan, θ)

function (::MeanToNatural{NormalGamma})(tuple_of_θ::Tuple{Any, Any, Any, Any})
    (μ, λ, α, β) = tuple_of_θ
    η1 = λ*μ
    η2 = -λ/2
    η3 = α - 1/2
    η4 = -β - λ*(μ^2)/2
    return (η1,η2,η3,η4)
end

function (::NaturalToMean{NormalGamma})(tuple_of_η::Tuple{Any, Any, Any, Any})
    (η1, η2, η3, η4) = tuple_of_η
    μ = η1 * (-1 / 2) / (η2)
    λ = -2η2
    α = η3 + (1 / 2)
    β = -η4 + (η1^2 / 4η2)

    return (μ, λ, α, β)
end

function unpack_parameters(::Type{NormalGamma}, packed)
    fi = firstindex(packed)
    return (packed[fi], packed[fi+1], packed[fi+2], packed[fi+3])
end

isbasemeasureconstant(::Type{NormalGamma}) = ConstantBaseMeasure()

getbasemeasure(::Type{NormalGamma}) = (x) ->  invsqrt2π
# x is a 2d vector where first dimension is mean and the second dimension is precision component
getsufficientstatistics(::Type{NormalGamma}) = (x -> x[1]*x[2], x -> x[2]*x[1]^2, x -> log(x[2]), x -> x[2])

getlogpartition(::NaturalParametersSpace, ::Type{NormalGamma}) = (η) -> begin
    (η1, η2, η3, η4) = unpack_parameters(NormalGamma, η)
    η3half = η3 + (1 / 2)
    return loggamma(η3half) - log(-2η2) * (1 / 2) - (η3half) * log(-η4 + η1^2 / (4η2))
end

getfisherinformation(::NaturalParametersSpace, ::Type{NormalGamma}) = (η) -> begin
    (η1, η2, η3, η4) = unpack_parameters(NormalGamma, η)
    # Define a 4x4 matrix
    info_matrix = Matrix{Float64}(undef, 4, 4)
    tmp1 = ((η1^2) / (4η2) - η4)^-1
    tmp2 = ((η1^2) / (4η2) - η4)^-2
    # Assign each value to the corresponding cell in the matrix
    info_matrix[1, 1] =
        ((tmp1) * ((-1 / 2) - η3) + (-(η1^2) * (tmp2) * ((-1 / 2) - η3)) / (2η2)) /
        (2η2)
    info_matrix[2, 1] =
        (-η1 * (tmp1) * ((-1 / 2) - η3)) / (2 * (η2^2)) +
        (2η1 * ((η1^2) / (16 * (η2^2))) * (tmp2) * ((-1 / 2) - η3)) / η2
    info_matrix[3, 1] = (-η1 * (tmp1)) / (2η2)
    info_matrix[4, 1] = (η1 * (tmp2) * ((-1 / 2) - η3)) / (2η2)
    info_matrix[1, 2] =
        ((η1^3) * (tmp2) * ((-1 / 2) - η3)) / (8 * (η2^3)) +
        (-η1 * (tmp1) * ((-1 / 2) - η3)) / (2 * (η2^2))
    info_matrix[2, 2] =
        (1 // 2) * (η2^-2) + (-(η1^2) * ((η1^2) / (16 * (η2^2))) * (tmp2) * ((-1 / 2) - η3)) / (η2^2) +
        128η2 * ((η1^2) / (256 * (η2^4))) * (tmp1) * ((-1 / 2) - η3)
    info_matrix[3, 2] = ((η1^2) * (tmp1)) / (4 * (η2^2))
    info_matrix[4, 2] = (-(η1^2) * (tmp2) * ((-1 / 2) - η3)) / (4 * (η2^2))
    info_matrix[1, 3] = (-η1 * (tmp1)) / (2η2)
    info_matrix[2, 3] = 4 * ((η1^2) / (16 * (η2^2))) * (tmp1)
    info_matrix[3, 3] = SpecialFunctions.trigamma((1 / 2) + η3)
    info_matrix[4, 3] = tmp1
    info_matrix[1, 4] = (-η1 * ((1 / 2) + η3) * (tmp2)) / (2η2)
    info_matrix[2, 4] = 4 * ((1 / 2) + η3) * ((η1^2) / (16 * (η2^2))) * (tmp2)
    info_matrix[3, 4] = tmp1
    info_matrix[4, 4] = ((1 / 2) + η3) * (tmp2)

    return info_matrix
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{NormalGamma}) = (θ) -> begin
    (_, λ, α, β) = unpack_parameters(NormalGamma, θ)
    return loggamma(α) - α*log(β) - (1/2)*log(λ)
end

getfisherinformation(::MeanParametersSpace, ::Type{NormalGamma}) = (θ) -> begin
    (_, λ, α, β) = unpack_parameters(NormalGamma, θ)
    info_matrix = Matrix{Float64}(undef, 4, 4)

    # Assign each value to the corresponding cell in the matrix
    info_matrix[1, 1] = -λ * α / β
    info_matrix[2, 1] = 0
    info_matrix[3, 1] = 0
    info_matrix[4, 1] = 0
    info_matrix[1, 2] = 0
    info_matrix[2, 2] = (-1 / 2) * (λ^-2)
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




















# sufficientstatistics(::Union{<:ExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}) =
#     (x, τ) -> SA[τ*x, τ*x^2, log(τ), τ]

# sufficientstatistics(union::Union{<:ExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}, x) =
#     sufficientstatistics(union)(x[1], x[2])

# function pack_naturalparameters(dist::NormalGamma)
#     μ, λ, α, β = params(dist)
#     η1 = λ * μ
#     η2 = -λ * (1 / 2)
#     η3 = α - (1 / 2)
#     η4 = -β - λ * μ^2 * (1 / 2)

#     return [η1, η2, η3, η4]
# end

# function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:NormalGamma})
#     η = getnaturalparameters(ef)
#     @inbounds η1 = η[1]
#     @inbounds η2 = η[2]
#     @inbounds η3 = η[3]
#     @inbounds η4 = η[4]

#     return η1, η2, η3, η4
# end

# Base.convert(::Type{ExponentialFamilyDistribution}, dist::NormalGamma) =
#     ExponentialFamilyDistribution(NormalGamma, pack_naturalparameters(dist))

# function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{NormalGamma})
#     η1, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)
#     return NormalGamma(η1 * (-1 / 2) / (η2), -2η2, η3 + (1 / 2), -η4 + (η1^2 / 4η2))
# end

# function logpartition(exponentialfamily::ExponentialFamilyDistribution{NormalGamma})
#     η1, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)
#     η3half = η3 + (1 / 2)
#     return loggamma(η3half) - log(-2η2) * (1 / 2) - (η3half) * log(-η4 + η1^2 / (4η2))
# end

# function isproper(exponentialfamily::ExponentialFamilyDistribution{NormalGamma})
#     _, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)
#     return -η2 > 0 && (η3 >= tiny + minushalf) && (-η4 >= tiny)
# end

# basemeasure(::Union{<:ExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}) = 1 / sqrt2π
# basemeasure(::Union{<:ExponentialFamilyDistribution{NormalGamma}, <:NormalGamma}, x) = 1 / sqrt2π

# function Random.rand!(rng::AbstractRNG, dist::NormalGamma, container::AbstractVector)
#     container[2] = rand(rng, GammaShapeRate(dist.α, dist.β))
#     container[1] = rand(rng, NormalMeanPrecision(dist.μ, dist.λ * container[2]))
#     return container
# end

# function Random.rand!(rng::AbstractRNG, dist::NormalGamma, container::AbstractVector{T}) where {T <: Vector}
#     for i in eachindex(container)
#         rand!(rng, dist, container[i])
#     end
#     return container
# end

# function Random.rand(rng::AbstractRNG, dist::NormalGamma)
#     container = Vector{Real}(undef, 2)
#     rand!(rng, dist, container)
#     return container
# end

# function Random.rand(rng::AbstractRNG, dist::NormalGamma, nsamples::Int)
#     container = Vector{Vector{Real}}(undef, nsamples)
#     for i in eachindex(container)
#         container[i] = Vector{Real}(undef, 2)
#         rand!(rng, dist, container[i])
#     end
#     return container
# end

# ##fisher information should be optimized further
# function fisherinformation(exponentialfamily::ExponentialFamilyDistribution{NormalGamma})
#     η1, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)

#     # Define a 4x4 matrix
#     info_matrix = Matrix{Float64}(undef, 4, 4)
#     tmp1 = ((η1^2) / (4η2) - η4)^-1
#     tmp2 = ((η1^2) / (4η2) - η4)^-2
#     # Assign each value to the corresponding cell in the matrix
#     info_matrix[1, 1] =
#         ((tmp1) * ((-1 / 2) - η3) + (-(η1^2) * (tmp2) * ((-1 / 2) - η3)) / (2η2)) /
#         (2η2)
#     info_matrix[2, 1] =
#         (-η1 * (tmp1) * ((-1 / 2) - η3)) / (2 * (η2^2)) +
#         (2η1 * ((η1^2) / (16 * (η2^2))) * (tmp2) * ((-1 / 2) - η3)) / η2
#     info_matrix[3, 1] = (-η1 * (tmp1)) / (2η2)
#     info_matrix[4, 1] = (η1 * (tmp2) * ((-1 / 2) - η3)) / (2η2)
#     info_matrix[1, 2] =
#         ((η1^3) * (tmp2) * ((-1 / 2) - η3)) / (8 * (η2^3)) +
#         (-η1 * (tmp1) * ((-1 / 2) - η3)) / (2 * (η2^2))
#     info_matrix[2, 2] =
#         (1 // 2) * (η2^-2) + (-(η1^2) * ((η1^2) / (16 * (η2^2))) * (tmp2) * ((-1 / 2) - η3)) / (η2^2) +
#         128η2 * ((η1^2) / (256 * (η2^4))) * (tmp1) * ((-1 / 2) - η3)
#     info_matrix[3, 2] = ((η1^2) * (tmp1)) / (4 * (η2^2))
#     info_matrix[4, 2] = (-(η1^2) * (tmp2) * ((-1 / 2) - η3)) / (4 * (η2^2))
#     info_matrix[1, 3] = (-η1 * (tmp1)) / (2η2)
#     info_matrix[2, 3] = 4 * ((η1^2) / (16 * (η2^2))) * (tmp1)
#     info_matrix[3, 3] = SpecialFunctions.trigamma((1 / 2) + η3)
#     info_matrix[4, 3] = tmp1
#     info_matrix[1, 4] = (-η1 * ((1 / 2) + η3) * (tmp2)) / (2η2)
#     info_matrix[2, 4] = 4 * ((1 / 2) + η3) * ((η1^2) / (16 * (η2^2))) * (tmp2)
#     info_matrix[3, 4] = tmp1
#     info_matrix[4, 4] = ((1 / 2) + η3) * (tmp2)

#     return info_matrix
# end

# ## Allocates for zeros
# function fisherinformation(dist::NormalGamma)
#     μ, λ, α, β = params(dist)

#     # Define a 4x4 matrix
#     info_matrix = Matrix{Float64}(undef, 4, 4)

#     # Assign each value to the corresponding cell in the matrix
#     info_matrix[1, 1] = -λ * α / β
#     info_matrix[2, 1] = 0
#     info_matrix[3, 1] = 0
#     info_matrix[4, 1] = 0
#     info_matrix[1, 2] = 0
#     info_matrix[2, 2] = (-1 / 2) * (λ^-2)
#     info_matrix[3, 2] = 0
#     info_matrix[4, 2] = 0
#     info_matrix[1, 3] = 0
#     info_matrix[2, 3] = 0
#     info_matrix[3, 3] = -SpecialFunctions.trigamma(α)
#     info_matrix[4, 3] = β^-1
#     info_matrix[1, 4] = 0
#     info_matrix[2, 4] = 0
#     info_matrix[3, 4] = β^-1
#     info_matrix[4, 4] = -α * (β^-2)

#     # Return the resulting information matrix
#     return -info_matrix
# end
