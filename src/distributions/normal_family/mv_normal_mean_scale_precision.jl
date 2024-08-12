export MvNormalMeanScalePrecision, MvGaussianMeanScalePrecision

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

"""
    MvNormalMeanScalePrecision{T <: Real, M <: AbstractVector{T}} <: AbstractMvNormal

A multivariate normal distribution with mean `μ` and scale parameter `γ` that scales the identity precision matrix.

# Type Parameters
- `T`: The element type of the mean vector and scale parameter
- `M`: The type of the mean vector, which must be a subtype of `AbstractVector{T}`

# Fields
- `μ::M`: The mean vector of the multivariate normal distribution
- `γ::T`: The scale parameter that scales the identity precision matrix

# Notes
The precision matrix of this distribution is `γ * I`, where `I` is the identity matrix.
The covariance matrix is the inverse of the precision matrix, i.e., `(1/γ) * I`.
"""
struct MvNormalMeanScalePrecision{T <: Real, M <: AbstractVector{T}} <: AbstractMvNormal
    μ::M
    γ::T
end

const MvGaussianMeanScalePrecision = MvNormalMeanScalePrecision

function MvNormalMeanScalePrecision(μ::AbstractVector{<:Real}, γ::Real)
    T = promote_type(eltype(μ), eltype(γ))
    return MvNormalMeanScalePrecision(convert(AbstractArray{T}, μ), convert(T, γ))
end

function MvNormalMeanScalePrecision(μ::AbstractVector{<:Integer}, γ::Real)
    return MvNormalMeanScalePrecision(float.(μ), float(γ))
end

function MvNormalMeanScalePrecision(μ::AbstractVector{T}) where {T}
    return MvNormalMeanScalePrecision(μ, convert(T, 1))
end

function MvNormalMeanScalePrecision(μ::AbstractVector{T1}, γ::T2) where {T1, T2}
    T = promote_type(T1, T2)
    μ_new = convert(AbstractArray{T}, μ)
    γ_new = convert(T, γ)(length(μ))
    return MvNormalMeanScalePrecision(μ_new, γ_new)
end

function unpack_parameters(::Type{MvNormalMeanScalePrecision}, packed)
    len = length(packed)
    n = div(-1 + isqrt(1 + 4 * len), 2)

    p₁ = view(packed, 1:n)
    p₂ = packed[end]

    return (p₁, p₂)
end

function isproper(::NaturalParametersSpace, ::Type{MvNormalMeanScalePrecision}, η, conditioner)
    k = length(η) - 1
    if length(η) < 2 || (length(η) !== k + 1)
        return false
    end
    (η₁, η₂) = unpack_parameters(MvNormalMeanScalePrecision, η)
    return isnothing(conditioner) && length(η₁) === size(η₂, 1) && (size(η₂, 1) === size(η₂, 2)) && isposdef(-η₂)
end

function (::MeanToNatural{MvNormalMeanScalePrecision})(tuple_of_θ::Tuple{Any, Any})
    (μ, γ) = tuple_of_θ
    Σ⁻¹ = 1 / γ
    return (Σ⁻¹ * μ, Σ⁻¹ / -2)
end

# Conversions
function Base.convert(
    ::Type{MvNormal{T, C, M}},
    dist::MvNormalMeanScalePrecision
) where {T <: Real, C <: Distributions.PDMats.PDMat{T, Matrix{T}}, M <: AbstractVector{T}}
    m, σ = mean(dist), std(dist)
    return MvNormal(convert(M, m), convert(T, σ))
end

function Base.convert(
    ::Type{MvNormalMeanScalePrecision{T, M}},
    dist::MvNormalMeanScalePrecision
) where {T <: Real, M <: AbstractArray{T}}
    m, γ = mean(dist), dist.γ
    return MvNormalMeanScalePrecision{T, M}(convert(M, m), convert(T, γ))
end

function Base.convert(
    ::Type{MvNormalMeanScalePrecision{T}},
    dist::MvNormalMeanScalePrecision
) where {T <: Real}
    return convert(MvNormalMeanScalePrecision{T, AbstractArray{T, 1}}, dist)
end

function Base.convert(::Type{MvNormalMeanCovariance}, dist::MvNormalMeanScalePrecision)
    m, σ = mean(dist), cov(dist)
    return MvNormalMeanCovariance(m, σ * diagm(ones(length(m))))
end

function Base.convert(::Type{MvNormalMeanPrecision}, dist::MvNormalMeanScalePrecision)
    m, γ = mean(dist), precision(dist)
    return MvNormalMeanPrecision(m, γ * diagm(ones(length(m))))
end

function Base.convert(::Type{MvNormalWeightedMeanPrecision}, dist::MvNormalMeanScalePrecision)
    m, γ = mean(dist), precision(dist)
    return MvNormalWeightedMeanPrecision(γ * m, γ * diagm(ones(length(m))))
end

Distributions.distrname(::MvNormalMeanScalePrecision) = "MvNormalMeanScalePrecision"

BayesBase.weightedmean(dist::MvNormalMeanScalePrecision) = precision(dist) * mean(dist)

BayesBase.mean(dist::MvNormalMeanScalePrecision)      = dist.μ
BayesBase.mode(dist::MvNormalMeanScalePrecision)      = mean(dist)
BayesBase.var(dist::MvNormalMeanScalePrecision)       = diag(cov(dist))
BayesBase.cov(dist::MvNormalMeanScalePrecision)       = cholinv(invcov(dist))
BayesBase.invcov(dist::MvNormalMeanScalePrecision)    = scale(dist) * I(length(mean(dist)))
BayesBase.std(dist::MvNormalMeanScalePrecision)       = cholsqrt(cov(dist))
BayesBase.logdetcov(dist::MvNormalMeanScalePrecision) = -chollogdet(invcov(dist))
BayesBase.scale(dist::MvNormalMeanScalePrecision)     = dist.γ
BayesBase.params(dist::MvNormalMeanScalePrecision)    = (mean(dist), scale(dist))

function Distributions.sqmahal(dist::MvNormalMeanScalePrecision, x::AbstractVector)
    T = promote_type(eltype(x), paramfloattype(dist))
    return sqmahal!(similar(x, T), dist, x)
end

function Distributions.sqmahal!(r, dist::MvNormalMeanScalePrecision, x::AbstractVector)
    μ = mean(dist)
    @inbounds @simd for i in 1:length(r)
        r[i] = μ[i] - x[i]
    end
    return dot3arg(r, invcov(dist), r) # x' * A * x
end

Base.eltype(::MvNormalMeanScalePrecision{T}) where {T} = T
Base.precision(dist::MvNormalMeanScalePrecision) = invcov(dist)
Base.length(dist::MvNormalMeanScalePrecision) = length(mean(dist))
Base.ndims(dist::MvNormalMeanScalePrecision) = length(dist)
Base.size(dist::MvNormalMeanScalePrecision) = (length(dist),)

Base.convert(::Type{<:MvNormalMeanScalePrecision}, μ::AbstractVector, γ::Real) = MvNormalMeanScalePrecision(μ, γ)

function Base.convert(::Type{<:MvNormalMeanScalePrecision{T}}, μ::AbstractVector, γ::T) where {T <: Real}
    MvNormalMeanScalePrecision(convert(AbstractArray{T}, μ), convert(T, γ))
end

BayesBase.vague(::Type{<:MvNormalMeanScalePrecision}, dims::Int) =
    MvNormalMeanScalePrecision(zeros(Float64, dims), convert(Float64, tiny))

BayesBase.default_prod_rule(::Type{<:MvNormalMeanScalePrecision}, ::Type{<:MvNormalMeanScalePrecision}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalMeanScalePrecision, right::MvNormalMeanScalePrecision)
    w = scale(left) + scale(right)
    m = (scale(left) * mean(left) + scale(right) * mean(right)) / w
    return MvNormalMeanScalePrecision(m, w)
end

BayesBase.default_prod_rule(::Type{<:MultivariateNormalDistributionsFamily}, ::Type{<:MvNormalMeanScalePrecision}) = PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::L,
    right::R
) where {L <: MultivariateNormalDistributionsFamily, R <: MvNormalMeanScalePrecision}
    wleft  = convert(MvNormalWeightedMeanPrecision, left)
    wright = convert(MvNormalWeightedMeanPrecision, right)
    return prod(BayesBase.default_prod_rule(wleft, wright), wleft, wright)
end
