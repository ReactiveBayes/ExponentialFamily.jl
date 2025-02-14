using StatsFuns: logistic
using StatsFuns: softmax, softmax!
using SpecialFunctions: gamma, loggamma
using Distributions

import ForwardDiff

# We create a specialized 3-argument dot function, because the built-in Julia version is not auto-differentiable
function dot3arg(x, A, y)
    return dot(x, A, y)
end

function dot3arg(x::AbstractVector, A::AbstractMatrix, y::AbstractVector{D}) where {D <: ForwardDiff.Dual}
    (axes(x)..., axes(y)...) == axes(A) || throw(DimensionMismatch())
    T = typeof(dot(first(x), first(A), first(y)))
    s = zero(T)
    i₁ = first(eachindex(x))
    x₁ = first(x)
    @inbounds for j in eachindex(y)
        yj = y[j]
        temp = zero(adjoint(A[i₁, j]) * x₁)
        @simd for i in eachindex(x)
            temp += adjoint(A[i, j]) * x[i]
        end
        s += dot(temp, yj)
    end
    return s
end

function binomial_prod(n, p, x)
    try
        b1 = binomial(n, x)
        b2 = binomial(p, x)
        result, flag = Base.mul_with_overflow(b1, b2)
        if flag
            return binomial_prod(big(n), big(p), big(x))
        else
            return result
        end
    catch error
        if isa(error, OverflowError)
            return binomial_prod(big(n), big(p), big(x))
        else
            throw(error)
        end
    end
end

mvdigamma(η, p) = sum(digamma(η + (one(d) - d) / 2) for d in 1:p)

abstract type VectorMatrixvariate <: VariateForm end
const VectorMatrixDistribution{S <: ValueSupport} = Distribution{VectorMatrixvariate, S}
const ContinuousMultivariateMatrixvariateDistribution = Distribution{VectorMatrixvariate, Continuous}
