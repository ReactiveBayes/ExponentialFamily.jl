export MvNormalWeightedMeanPrecision

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod
"""
    MvNormalWeightedMeanPrecision{T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T}} <: AbstractMvNormal

A multivariate normal distribution with a weighted mean vector `xi` and precision matrix `Λ`, where `T` is the element type of the vectors `M` and matrices `P`. This struct represents a natural parametrization of a multivariate Gaussian distribution.

# Fields
- `xi::M`: The weighted mean vector of the multivariate normal distribution.
- `Λ::P`: The precision matrix (inverse of the covariance matrix) of the multivariate normal distribution.
"""
struct MvNormalWeightedMeanPrecision{T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T}} <: AbstractMvNormal
    xi :: M
    Λ  :: P
end

function MvNormalWeightedMeanPrecision(xi::AbstractVector{<:Real}, Λ::AbstractMatrix{<:Real})
    T = promote_type(eltype(xi), eltype(Λ))
    return MvNormalWeightedMeanPrecision(convert(AbstractArray{T}, xi), convert(AbstractArray{T}, Λ))
end

function MvNormalWeightedMeanPrecision(xi::AbstractVector{<:Integer}, Λ::AbstractMatrix{<:Integer})
    return MvNormalWeightedMeanPrecision(float.(xi), float.(Λ))
end

function MvNormalWeightedMeanPrecision(xi::AbstractVector{L}, λ::AbstractVector{R}) where {L, R}
    return MvNormalWeightedMeanPrecision(xi, convert(Matrix{promote_type(L, R)}, Diagonal(λ)))
end

function MvNormalWeightedMeanPrecision(xi::AbstractVector{T}) where {T}
    return MvNormalWeightedMeanPrecision(xi, convert(AbstractArray{T}, ones(length(xi))))
end

Distributions.distrname(::MvNormalWeightedMeanPrecision) = "MvNormalWeightedMeanPrecision"

BayesBase.weightedmean(dist::MvNormalWeightedMeanPrecision) = dist.xi

function BayesBase.mean_cov(dist::MvNormalWeightedMeanPrecision)
    Σ = cov(dist)
    μ = Σ * weightedmean(dist)
    return (μ, Σ)
end

function BayesBase.mean(dist::MvNormalWeightedMeanPrecision)
    z = fastcholesky(precision(dist))
    return z \ weightedmean(dist)
end
BayesBase.mode(dist::MvNormalWeightedMeanPrecision)      = mean(dist)
BayesBase.var(dist::MvNormalWeightedMeanPrecision)       = diag(cov(dist))
BayesBase.cov(dist::MvNormalWeightedMeanPrecision)       = cholinv(dist.Λ)
BayesBase.invcov(dist::MvNormalWeightedMeanPrecision)    = dist.Λ
BayesBase.std(dist::MvNormalWeightedMeanPrecision)       = cholsqrt(cov(dist))
BayesBase.logdetcov(dist::MvNormalWeightedMeanPrecision) = -chollogdet(invcov(dist))
BayesBase.params(dist::MvNormalWeightedMeanPrecision)    = (weightedmean(dist), invcov(dist))
diagonal_skewness(dist::MvNormalWeightedMeanPrecision)   = zeros(length(dist.μ))
diagonal_kurtosis(dist::MvNormalWeightedMeanPrecision)   = 3*var(dist).^2
function Distributions.sqmahal(dist::MvNormalWeightedMeanPrecision, x::AbstractVector)
    T = promote_type(eltype(x), paramfloattype(dist))
    return sqmahal!(similar(x, T), dist, x)
end

function Distributions.sqmahal!(r, dist::MvNormalWeightedMeanPrecision, x::AbstractVector)
    μ = mean(dist)
    @inbounds @simd for i in 1:length(r)
        r[i] = μ[i] - x[i]
    end
    return dot3arg(r, invcov(dist), r) # x' * A * x
end

Base.eltype(::MvNormalWeightedMeanPrecision{T}) where {T} = T
Base.precision(dist::MvNormalWeightedMeanPrecision)       = invcov(dist)
Base.length(dist::MvNormalWeightedMeanPrecision)          = length(weightedmean(dist))
Base.ndims(dist::MvNormalWeightedMeanPrecision)           = length(dist)
Base.size(dist::MvNormalWeightedMeanPrecision)            = (length(dist),)

Base.convert(::Type{<:MvNormalWeightedMeanPrecision}, xi::AbstractVector, Λ::AbstractMatrix) =
    MvNormalWeightedMeanPrecision(xi, Λ)

function Base.convert(
    ::Type{<:MvNormalWeightedMeanPrecision{T}},
    xi::AbstractVector,
    Λ::AbstractMatrix
) where {T <: Real}
    MvNormalWeightedMeanPrecision(convert(AbstractArray{T}, xi), convert(AbstractArray{T}, Λ))
end

BayesBase.vague(::Type{<:MvNormalWeightedMeanPrecision}, dims::Int) =
    MvNormalWeightedMeanPrecision(zeros(Float64, dims), fill(convert(Float64, tiny), dims))

BayesBase.default_prod_rule(::Type{<:MvNormalWeightedMeanPrecision}, ::Type{<:MvNormalWeightedMeanPrecision}) =
    PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalWeightedMeanPrecision, right::MvNormalWeightedMeanPrecision)
    xi = weightedmean(left) + weightedmean(right)
    Λ  = invcov(left) + invcov(right)
    return MvNormalWeightedMeanPrecision(xi, Λ)
end
