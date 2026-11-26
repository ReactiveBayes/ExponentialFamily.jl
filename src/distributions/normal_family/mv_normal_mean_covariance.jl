export MvNormalMeanCovariance

import Distributions: logdetcov, sqmahal, sqmahal!, AbstractMvNormal
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

function MvNormalMeanCovariance(μ::AbstractVector{T1}, Σ::UniformScaling{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    μ_new = convert(AbstractArray{T}, μ)
    Σ_new = convert(UniformScaling{T}, Σ)(length(μ))
    return MvNormalMeanCovariance(μ_new, Σ_new)
end

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

# The product is computed in covariance form, which inverts only `Σₗ + Σᵣ`:
#
#     K = Σₗ (Σₗ + Σᵣ)⁻¹,    μ = μₗ + K (μᵣ - μₗ),    Σ = K Σᵣ.
#
# It stays exact when one of the covariances is singular, as is the message out of a linear map
# to a higher dimension (`z = A * x` with `A` taller than wide). Such a message has no precision
# matrix, and a product in precision form returns a wrong mean and covariance. `K Σᵣ` is used
# instead of `Σₗ - K Σₗ`, which cancels catastrophically when `Σₗ` is vague.
function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalMeanCovariance, right::MvNormalMeanCovariance)
    μ_left, Σ_left = mean_cov(left)
    μ_right, Σ_right = mean_cov(right)
    K = (fastcholesky(Σ_left + Σ_right) \ Σ_left)' # Σₗ (Σₗ + Σᵣ)⁻¹, as Σₗ is symmetric
    μ = μ_left + K * (μ_right - μ_left)
    Σ = K * Σ_right
    return MvNormalMeanCovariance(μ, (Σ + Σ') / 2)
end

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalMeanCovariance{T, <:Vector, <:Matrix},
    right::MvNormalMeanCovariance{T, <:Vector, <:Matrix}
) where {T <: LinearAlgebra.BlasFloat}
    μ_left, Σ_left = mean_cov(left)
    μ_right, Σ_right = mean_cov(right)
    LinearAlgebra.checksquare(Σ_left) == LinearAlgebra.checksquare(Σ_right) == length(μ_left) == length(μ_right) ||
        throw(DimensionMismatch("cannot multiply Gaussians of dimensions $(length(μ_left)) and $(length(μ_right))"))

    # G = (Σₗ + Σᵣ)⁻¹ Σₗ = Kᵀ
    G = LinearAlgebra.ldiv!(fastcholesky!(Σ_left + Σ_right), copy(Σ_left))

    # μ = μₗ + Gᵀ (μᵣ - μₗ)
    μ = BLAS.gemv!('T', one(T), G, μ_right - μ_left, one(T), copy(μ_left))

    # Σ = Gᵀ Σᵣ, symmetrized in place
    Σ = BLAS.gemm('T', 'N', G, Σ_right)
    n = size(Σ, 1)
    @inbounds for j in 1:n, i in (j+1):n
        Σ[i, j] = Σ[j, i] = (Σ[i, j] + Σ[j, i]) / 2
    end

    return MvNormalMeanCovariance(μ, Σ)
end
