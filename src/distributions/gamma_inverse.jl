export GammaInverse
import Distributions: InverseGamma, shape, scale
import SpecialFunctions: digamma

const GammaInverse = InverseGamma

vague(::Type{<:GammaInverse}) = InverseGamma(2.0, huge)

closed_prod_rule(::Type{<:GammaInverse}, ::Type{<:GammaInverse}) = ClosedProd()

function Base.prod(::ClosedProd, left::GammaInverse, right::InverseGamma)
    return GammaInverse(shape(left) + shape(right) + one(Float64), scale(left) + scale(right))
end

function mean(::typeof(log), dist::GammaInverse)
    α = shape(dist)
    θ = scale(dist)
    return log(θ) - digamma(α)
end

function mean(::typeof(inv), dist::GammaInverse)
    α = shape(dist)
    θ = scale(dist)
    return α / θ
end
