
import BayesBase
using LinearAlgebra
using Distributions
using RxInfer
using SpecialFunctions



struct MvLocationScaleT{T, N <: Real, M <: AbstractVector{T}, S <: AbstractMatrix{T}} <: ContinuousMultivariateDistribution
 
    η::N # Degrees-of-freedom
    μ::M # Mean vector
    Σ::S # Covariance matrix

    function MvLocationScaleT(η::N, μ::M, Σ::S) where {T, N <: Real, M <: AbstractVector{T}, S <: AbstractMatrix{T}}

        dims = length(μ)
        if η <= dims; error("Degrees of freedom parameter must be larger than the dimensionality."); end
        if dims !== size(Σ,1); error("Dimensionalities of mean and covariance matrix don't match."); end

        return new{T,N,M,S}(η, μ, Σ)
    end
end

BayesBase.params(p::MvLocationScaleT) = (p.η, p.μ, p.Σ)
BayesBase.ndims(p::MvLocationScaleT) = length(p.μ)
BayesBase.mean(p::MvLocationScaleT) = p.μ
BayesBase.mode(p::MvLocationScaleT) = p.μ
BayesBase.cov(p::MvLocationScaleT) = p.η > 2 ? p.η/(p.η-2)*p.Σ : error("Degrees of freedom parameter must be larger than 2.")
BayesBase.precision(p::MvLocationScaleT) = inv(cov(p))

function pdf(p::MvLocationScaleT, x::Vector)
    d = ndims(p)
    η, μ, Σ = params(p)
    return sqrt(1/( (η*π)^d*det(Σ) )) * gamma((η+d)/2)/gamma(η/2) * (1 + 1/η*(x-μ)'*inv(Σ)*(x-μ))^(-(η+d)/2)
end

function Distributions.logpdf(p::MvLocationScaleT, x::Vector)
    d = ndims(p)
    η, μ, Σ = params(p)
    return -d/2*log(η*π) - 1/2*logdet(Σ) +loggamma((η+d)/2) -loggamma(η/2) -(η+d)/2*log(1 + 1/η*(x-μ)'*inv(Σ)*(x-μ))
end

BayesBase.default_prod_rule(::Type{<:MvLocationScaleT}, ::Type{<:MvLocationScaleT}) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:AbstractMvNormal}, ::Type{<:MvLocationScaleT}) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:MvLocationScaleT}, ::Type{<:AbstractMvNormal}) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:MvLocationScaleT}, ::Type{<:unBoltzmann}) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:unBoltzmann}, ::Type{<:MvLocationScaleT}) = BayesBase.ClosedProd()

function BayesBase.prod(::BayesBase.ClosedProd, left::MvLocationScaleT, right::MvLocationScaleT)    
    if ndims(left) != ndims(right); error("Dimensionalities of MvLocationScaleT dists are not the same."); end

    ηl,μl,Σl = params(left)
    ηr,μr,Σr = params(right)

    Λl = inv(ηl/(ηl-2)*Σl)
    Λr = inv(ηr/(ηr-2)*Σr)
    
    Σ = inv( Λl + Λr )
    μ = Σ*(Λl*μl + Λr*μr)
    
    return MvNormalMeanCovariance(μ,Σ)
end

function BayesBase.prod(::BayesBase.ClosedProd, left::AbstractMvNormal, right::MvLocationScaleT)    
    if ndims(left) != ndims(right); error("Dimensionalities of MvLocationScaleT and MvNormal dists are not the same."); end

    μl,Σl = mean_cov(left)
    ηr,μr,Σr = params(right)

    Λl = inv(Σl)
    Λr = inv(ηr/(ηr-2)*Σr)
    
    Σ = inv( Λl + Λr )
    μ = Σ*(Λl*μl + Λr*μr)
    
    return MvNormalMeanCovariance(μ,Σ)
end

BayesBase.prod(::BayesBase.ClosedProd, left::MvLocationScaleT, right::AbstractMvNormal) = BayesBase.prod(ClosedProd(), right, left)