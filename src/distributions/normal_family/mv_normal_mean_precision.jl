export MvNormalMeanPrecision

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

"""
    MvNormalMeanPrecision{T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T}} <: AbstractMvNormal

A multivariate normal distribution with mean `μ` and precision matrix `Λ`, where `T` is the element type of the vectors `M` and matrices `P`.

# Fields
- `μ::M`: The mean vector of the multivariate normal distribution.
- `Λ::P`: The precision matrix (inverse of the covariance matrix) of the multivariate normal distribution.
"""
struct MvNormalMeanPrecision{T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T}} <: AbstractMvNormal
    μ::M
    Λ::P
end

function MvNormalMeanPrecision(μ::AbstractVector{<:Real}, Λ::AbstractMatrix{<:Real})
    T = promote_type(eltype(μ), eltype(Λ))
    return MvNormalMeanPrecision(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Λ))
end

function MvNormalMeanPrecision(μ::AbstractVector{<:Integer}, Λ::AbstractMatrix{<:Integer})
    return MvNormalMeanPrecision(float.(μ), float.(Λ))
end

function MvNormalMeanPrecision(μ::AbstractVector{L}, λ::AbstractVector{R}) where {L, R}
    return MvNormalMeanPrecision(μ, convert(Matrix{promote_type(L, R)}, Diagonal(λ)))
end

function MvNormalMeanPrecision(μ::AbstractVector{T}) where {T}
    return MvNormalMeanPrecision(μ, convert(AbstractArray{T}, ones(length(μ))))
end

function MvNormalMeanPrecision(μ::AbstractVector{T1}, Λ::UniformScaling{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    μ_new = convert(AbstractArray{T}, μ)
    Λ_new = convert(UniformScaling{T}, Λ)(length(μ))
    return MvNormalMeanPrecision(μ_new, Λ_new)
end

Distributions.distrname(::MvNormalMeanPrecision) = "MvNormalMeanPrecision"

BayesBase.weightedmean(dist::MvNormalMeanPrecision) = precision(dist) * mean(dist)

BayesBase.mean(dist::MvNormalMeanPrecision)      = dist.μ
BayesBase.mode(dist::MvNormalMeanPrecision)      = mean(dist)
BayesBase.var(dist::MvNormalMeanPrecision)       = diag(cov(dist))
BayesBase.cov(dist::MvNormalMeanPrecision)       = cholinv(dist.Λ)
BayesBase.invcov(dist::MvNormalMeanPrecision)    = dist.Λ
BayesBase.std(dist::MvNormalMeanPrecision)       = cholsqrt(cov(dist))
BayesBase.logdetcov(dist::MvNormalMeanPrecision) = -chollogdet(invcov(dist))
BayesBase.params(dist::MvNormalMeanPrecision)    = (mean(dist), invcov(dist))

function Distributions.sqmahal(dist::MvNormalMeanPrecision, x::AbstractVector)
    T = promote_type(eltype(x), paramfloattype(dist))
    return sqmahal!(similar(x, T), dist, x)
end

function Distributions.sqmahal!(r, dist::MvNormalMeanPrecision, x::AbstractVector)
    μ = mean(dist)
    @inbounds @simd for i in 1:length(r)
        r[i] = μ[i] - x[i]
    end
    return dot3arg(r, invcov(dist), r) # x' * A * x
end

Base.eltype(::MvNormalMeanPrecision{T}) where {T} = T
Base.precision(dist::MvNormalMeanPrecision)       = invcov(dist)
Base.length(dist::MvNormalMeanPrecision)          = length(mean(dist))
Base.ndims(dist::MvNormalMeanPrecision)           = length(dist)
Base.size(dist::MvNormalMeanPrecision)            = (length(dist),)

Base.convert(::Type{<:MvNormalMeanPrecision}, μ::AbstractVector, Λ::AbstractMatrix) = MvNormalMeanPrecision(μ, Λ)

function Base.convert(::Type{<:MvNormalMeanPrecision{T}}, μ::AbstractVector, Λ::AbstractMatrix) where {T <: Real}
    MvNormalMeanPrecision(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Λ))
end

BayesBase.vague(::Type{<:MvNormalMeanPrecision}, dims::Int) =
    MvNormalMeanPrecision(zeros(Float64, dims), fill(convert(Float64, tiny), dims))

BayesBase.default_prod_rule(::Type{<:MvNormalMeanPrecision}, ::Type{<:MvNormalMeanPrecision}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalMeanPrecision, right::MvNormalMeanPrecision)
    W = precision(left) + precision(right)
    xi = weightedmean(left) + weightedmean(right)
    return MvNormalWeightedMeanPrecision(xi, W)
end

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalMeanPrecision{T, <:Vector, <:Matrix},
    right::MvNormalMeanPrecision{T, <:Vector, <:Matrix}
) where {T <: LinearAlgebra.BlasFloat}
    W = precision(left) + precision(right)

    xi = weightedmean(right)

    xi = BLAS.gemv!('N', one(T), precision(left), mean(left), one(T), xi)

    return MvNormalWeightedMeanPrecision(xi, W)
end
