export MvNormalMeanScalePrecision

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
struct MvNormalMeanScalePrecision{T <: Real, M <: AbstractVector{T}}
    μ::M
    γ::T
end

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

Distributions.distrname(::MvNormalMeanScalePrecision) = "MvNormalMeanScalePrecision"

BayesBase.weightedmean(dist::MvNormalMeanScalePrecision) = precision(dist) * mean(dist)

BayesBase.mean(dist::MvNormalMeanScalePrecision)      = dist.μ
BayesBase.mode(dist::MvNormalMeanScalePrecision)      = mean(dist)
BayesBase.var(dist::MvNormalMeanScalePrecision)       = diag(cov(dist))
BayesBase.cov(dist::MvNormalMeanScalePrecision)       = cholinv(invcov(dist))
BayesBase.invcov(dist::MvNormalMeanScalePrecision)    = dist.γ * I(length(mean(dist)))
BayesBase.std(dist::MvNormalMeanScalePrecision)       = cholsqrt(cov(dist))
BayesBase.logdetcov(dist::MvNormalMeanScalePrecision) = -chollogdet(invcov(dist))
BayesBase.params(dist::MvNormalMeanScalePrecision)    = (mean(dist), invcov(dist))

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

# Base.convert(::Type{<:MvNormalMeanScalePrecision}, μ::AbstractVector, γ::Real) = MvNormalMeanScalePrecision(μ, γ)

# function Base.convert(::Type{<:MvNormalMeanScalePrecision{T}}, μ::AbstractVector, γ::T) where {T <: Real}
#     MvNormalMeanScalePrecision(convert(AbstractArray{T}, μ), convert(T, γ))
# end

BayesBase.vague(::Type{<:MvNormalMeanScalePrecision}, dims::Int) =
    MvNormalMeanScalePrecision(zeros(Float64, dims), convert(Float64, tiny))

BayesBase.default_prod_rule(::Type{<:MvNormalMeanScalePrecision}, ::Type{<:MvNormalMeanScalePrecision}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalMeanScalePrecision, right::MvNormalMeanScalePrecision)
    w = precision(left) + precision(right)
    m = (precision(left) * mean(left) + precision(right) * mean(right)) / w
    return MvNormalMeanScalePrecision(m, w)
end

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalMeanScalePrecision{T1},
    right::MvNormalMeanScalePrecision{T2}
) where {T1 <: LinearAlgebra.BlasFloat, T2 <: LinearAlgebra.BlasFloat}
    w = precision(left) + precision(right)

    xi = precision(right) * mean(right)
    T  = promote_type(T1, T2)
    xi = convert(AbstractVector{T}, xi)
    w  = convert(T, w)
    xi = BLAS.gemv!('N', one(T), convert(AbstractMatrix{T}, precision(left)), convert(AbstractVector{T}, mean(left)), one(T), xi)

    return MvNormalMeanScalePrecision(xi / w, w)
end
