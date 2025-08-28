
import BayesBase
using LinearAlgebra
using Distributions
using RxInfer
using SpecialFunctions


struct LocationScaleT <: ContinuousUnivariateDistribution
 
    ν  ::Real # Degrees-of-freedom
    μ  ::Real # Location parameter 
    σ  ::Real # Scale parameter (standard deviation)

    function LocationScaleT(ν::Float64, μ::Float64, σ::Float64)
        
        if ν <= 0.0; error("Degrees of freedom parameter must be positive."); end
        if σ <= 0.0; error("Standard deviation parameter must be positive."); end

        return new(ν, μ, σ)
    end
end

BayesBase.params(p::LocationScaleT) = (p.ν, p.μ, p.σ)
BayesBase.dim(p::LocationScaleT) = length(p.μ)
BayesBase.mean(p::LocationScaleT) = p.μ
BayesBase.std(p::LocationScaleT) = sqrt(p.ν/(p.ν-2))*p.σ
BayesBase.var(p::LocationScaleT) = p.ν/(p.ν-2)*p.σ^2
BayesBase.precision(p::LocationScaleT) = inv(var(p))

function pdf(p::LocationScaleT, x)
    ν, μ, σ = params(p)
    return gamma( (ν+1)/2 ) / ( gamma(ν/2) *sqrt(π*ν)*σ ) * ( 1 + (x-μ)^2/(ν*σ^2) )^( -(ν+1)/2 )
end

function logpdf(p::LocationScaleT, x)
    ν, μ, σ = params(p)
    return loggamma( (ν+1)/2 ) - loggamma(ν/2) - 1/2*log(π*ν) - log(σ) + ( -(ν+1)/2 )*log( 1 + (x-μ)^2/(ν*σ^2) )
end
