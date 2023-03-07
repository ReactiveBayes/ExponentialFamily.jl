import SpecialFunctions: loggamma
using Distributions

"""
    ν(x) ∝ exp(p*β*x - p*logГ(x)) ≡ exp(γ*x - p*logГ(x))
"""
struct GammaShapeLikelihood{T <: Real} <: ContinuousUnivariateDistribution
    p::T
    γ::T # p * β
end

Distributions.@distr_support GammaShapeLikelihood 0 Inf

Distributions.support(dist::GammaShapeLikelihood) = Distributions.RealInterval(minimum(dist), maximum(dist))

Base.show(io::IO, distribution::GammaShapeLikelihood{T}) where {T} =
    print(io, "GammaShapeLikelihood{$T}(π = $(distribution.p), γ = $(distribution.γ))")

Distributions.logpdf(distribution::GammaShapeLikelihood, x::Real) = distribution.γ * x - distribution.p * loggamma(x)

prod_analytical_rule(::Type{<:GammaShapeLikelihood}, ::Type{<:GammaShapeLikelihood}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::GammaShapeLikelihood, right::GammaShapeLikelihood)
    return GammaShapeLikelihood(left.p + right.p, left.γ + right.γ)
end
