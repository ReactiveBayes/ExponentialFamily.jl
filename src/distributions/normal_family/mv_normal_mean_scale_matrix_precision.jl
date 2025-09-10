export MvNormalMeanScaleMatrixPrecision

import Distributions: AbstractMvNormal



"""
    MvNormalMeanScaleMatrixPrecision{T <: Real, M <: AbstractVector{T}, H <: AbstractMatrix{T}} <: AbstractMvNormal

Multivariate normal with parameters (μ, γ, G) where the precision matrix Λ = γ * G.

At current state of development, this distribution simply returns a MvNormalMeanPrecision distribution.
This means that after creation, it is not possible to recover the original parameters, such as `γ` and `G`.

Requirements:
* γ > 0
* G is symmetric positive definite (SPD)
* tr(G) = d (enforced by normalisation)
* (equivalently Λ SPD)
"""
struct MvNormalMeanScaleMatrixPrecision{ T <: Real, M <: AbstractVector{T}, H <: AbstractMatrix{T}} <: AbstractMvNormal
    μ::M
    γ::T
    G::H
end

function MvNormalMeanScaleMatrixPrecision(μ::AbstractVector{<:Real}, γ::Real, G::AbstractMatrix{<:Real})
    @assert size(G,1) == size(G,2) "G must be square"
    @assert γ > 0.0 "γ must be positive"
    @assert isposdef(Symmetric(G)) "G must be SPD"
    T = promote_type(eltype(μ), eltype(γ), eltype(G))
    μc = convert(AbstractArray{T}, μ)
    γc = convert(T, γ)
    Gc = convert(AbstractMatrix{T}, G)
    return MvNormalMeanPrecision(μc, γc * Gc)
end

function MvNormalMeanScaleMatrixPrecision(μ::AbstractVector{<:Integer}, γ::Real, G::AbstractMatrix{<:Integer})
    @assert size(G,1) == size(G,2) "G must be square"
    @assert γ > 0.0 "γ must be positive"
    @assert isposdef(Symmetric(G)) "G must be SPD"
    return MvNormalMeanPrecision(float.(μ), float(γ) * float.(G))
end

function MvNormalMeanScaleMatrixPrecision(μ::AbstractVector{T}) where {T}
    return MvNormalMeanPrecision(μ, Matrix{T}(I, length(μ), length(μ)))
end

function MvNormalMeanScaleMatrixPrecision(μ::AbstractVector{<:T}, γ::T) where {T}
    @assert γ > zero(T) "γ must be positive"
    d = size(μ, 1)
    G = convert(AbstractMatrix{T}, Matrix{T}(I, d, d))
    return MvNormalMeanPrecision(μ, γ * G)
end

function MvNormalMeanScaleMatrixPrecision(μ::AbstractVector{T1}, γ::T2, G::AbstractMatrix{T3}) where {T1, T2, T3}
    @assert size(G,1) == size(G,2) "G must be square"
    @assert γ > zero(T2) "γ must be positive"
    @assert isposdef(Symmetric(G)) "G must be SPD"
    T = promote_type(T1, T2, T3)
    μ_new = convert(AbstractArray{T}, μ)
    γ_new = convert(T, γ)
    G_new = convert(AbstractMatrix{T}, G)
    return MvNormalMeanPrecision(μ_new, γ_new * G_new)
end
