using StatsFuns: logistic
using StatsFuns: softmax, softmax!
using SpecialFunctions: gamma, loggamma

import ForwardDiff

"""
    mirrorlog(x)

Returns `log(1 - x)`.
"""
mirrorlog(x) = log(one(x) - x)

"""
    xtlog(x)

Returns `x * log(x)`.
"""
xtlog(x) = x * log(x)

"""
    logmvbeta(x)

Uses the numerically stable algorithm to compute the logarithm of the multivariate beta distribution over with the parameter vector x.
"""
logmvbeta(x) = sum(loggamma, x) - loggamma(sum(x))

"""
    clamplog(x)

Same as `log` but clamps the input argument `x` to be in the range `tiny <= x <= typemax(x)` such that `log(0)` does not explode.
"""
clamplog(x) = log(clamp(x, tiny, typemax(x)))

"""
    mvtrigamma(p, x)

Computes multivariate trigamma function .
"""
mvtrigamma(p, x) = sum(trigamma(x + (one(x) - i) / 2) for i in 1:p)

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
