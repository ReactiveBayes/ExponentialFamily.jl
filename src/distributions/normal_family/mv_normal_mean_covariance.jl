export MvNormalMeanCovariance

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

"""
    MvNormalMeanCovariance{T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T}} <: AbstractMvNormal

A multivariate normal distribution with mean `μ` and covariance matrix `Σ`, where `T` is the element type of the vectors `M` and matrices `P`.

# Fields
- `μ::M`: The mean vector of the multivariate normal distribution.
- `Σ::P`: The covariance matrix of the multivariate normal distribution
"""
struct MvNormalMeanCovariance{T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T}} <: AbstractMvNormal
    μ::M
    Σ::P
end

function MvNormalMeanCovariance(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
    T = promote_type(eltype(μ), eltype(Σ))
    return MvNormalMeanCovariance(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

function MvNormalMeanCovariance(μ::AbstractVector{<:Integer}, Σ::AbstractMatrix{<:Integer})
    return MvNormalMeanCovariance(float.(μ), float.(Σ))
end

function MvNormalMeanCovariance(μ::AbstractVector{L}, σ::AbstractVector{R}) where {L, R}
    return MvNormalMeanCovariance(μ, convert(Matrix{promote_type(L, R)}, Diagonal(σ)))
end

function MvNormalMeanCovariance(μ::AbstractVector{T}) where {T}
    return MvNormalMeanCovariance(μ, convert(AbstractArray{T}, ones(length(μ))))
end

Distributions.distrname(::MvNormalMeanCovariance) = "MvNormalMeanCovariance"

function BayesBase.weightedmean(dist::MvNormalMeanCovariance)
    z = fastcholesky(cov(dist))
    return z \ mean(dist)
end

function BayesBase.weightedmean_invcov(dist::MvNormalMeanCovariance)
    W = precision(dist)
    xi = W * mean(dist)
    return (xi, W)
end

BayesBase.weightedmean_precision(dist::MvNormalMeanCovariance) = weightedmean_invcov(dist)

BayesBase.mean(dist::MvNormalMeanCovariance)      = dist.μ
BayesBase.var(dist::MvNormalMeanCovariance)       = diag(cov(dist))
BayesBase.cov(dist::MvNormalMeanCovariance)       = dist.Σ
BayesBase.invcov(dist::MvNormalMeanCovariance)    = cholinv(dist.Σ)
BayesBase.std(dist::MvNormalMeanCovariance)       = cholsqrt(cov(dist))
BayesBase.logdetcov(dist::MvNormalMeanCovariance) = chollogdet(cov(dist))
BayesBase.params(dist::MvNormalMeanCovariance)    = (mean(dist), cov(dist))
BayesBase.diagonal_skewness(dist::MvNormalMeanCovariance) = zeros(length(dist.μ))
BayesBase.diagonal_kurtosis(dist::MvNormalMeanCovariance) = 3*var(dist).^2

function Distributions.sqmahal(dist::MvNormalMeanCovariance, x::AbstractVector)
    T = promote_type(eltype(x), paramfloattype(dist))
    return sqmahal!(similar(x, T), dist, x)
end

function Distributions.sqmahal!(r, dist::MvNormalMeanCovariance, x::AbstractVector)
    μ = mean(dist)
    @inbounds @simd for i in 1:length(r)
        r[i] = μ[i] - x[i]
    end
    return dot3arg(r, invcov(dist), r) # x' * A * x
end

Base.eltype(::MvNormalMeanCovariance{T}) where {T} = T
Base.precision(dist::MvNormalMeanCovariance)       = invcov(dist)
Base.length(dist::MvNormalMeanCovariance)          = length(mean(dist))
Base.ndims(dist::MvNormalMeanCovariance)           = length(dist)
Base.size(dist::MvNormalMeanCovariance)            = (length(dist),)

Base.convert(::Type{<:MvNormalMeanCovariance}, μ::AbstractVector, Σ::AbstractMatrix) = MvNormalMeanCovariance(μ, Σ)

function Base.convert(::Type{<:MvNormalMeanCovariance{T}}, μ::AbstractVector, Σ::AbstractMatrix) where {T <: Real}
    return MvNormalMeanCovariance(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

BayesBase.vague(::Type{<:MvNormalMeanCovariance}, dims::Int) =
    MvNormalMeanCovariance(zeros(Float64, dims), fill(convert(Float64, huge), dims))

BayesBase.default_prod_rule(::Type{<:MvNormalMeanCovariance}, ::Type{<:MvNormalMeanCovariance}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalMeanCovariance, right::MvNormalMeanCovariance)
    xi_left, W_left = weightedmean_precision(left)
    xi_right, W_right = weightedmean_precision(right)
    return MvNormalWeightedMeanPrecision(xi_left + xi_right, W_left + W_right)
end

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalMeanCovariance{T1},
    right::MvNormalMeanCovariance{T2}
) where {T1 <: LinearAlgebra.BlasFloat, T2 <: LinearAlgebra.BlasFloat}
    xi, W = weightedmean_precision(left)

    W_right = precision(right)
    W .+= W_right

    T  = promote_type(T1, T2)
    xi = convert(AbstractVector{T}, xi)
    W  = convert(AbstractMatrix{T}, W)
    xi = BLAS.gemv!('N', one(T), convert(AbstractMatrix{T}, W_right), convert(AbstractVector{T}, mean(right)), one(T), xi)

    return MvNormalWeightedMeanPrecision(xi, W)
end
