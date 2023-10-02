export diageye

using StatsFuns: logistic
using StatsFuns: softmax, softmax!
using LoopVectorization
using SpecialFunctions: gamma, loggamma

import LinearAlgebra
import Base: show, maximum
import Base: convert, promote_rule

"""
    diageye(::Type{T}, n::Int)

An alias for the `Matrix{T}(I, n, n)`. Returns a matrix of size `n x n` with ones (of type `T`) on the diagonal and zeros everywhere else.
"""
diageye(::Type{T}, n::Int) where {T <: Real} = Matrix{T}(I, n, n)

"""
    diageye(n::Int)

An alias for the `Matrix{Float64}(I, n, n)`. Returns a matrix of size `n x n` with ones (of type `Float64`) on the diagonal and zeros everywhere else.
"""
diageye(n::Int) = diageye(Float64, n)

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
Same as `log` but clamps the input argument `x` to be in the range `tiny <= x <= typemax(x)` such that `log(0)` does not explode.
"""
clamplog(x) = log(clamp(x, tiny, typemax(x)))

"""
    mvtrigamma(p, x)

Computes multivariate trigamma function .
"""
mvtrigamma(p, x) = sum(trigamma(x + (one(x) - i) / 2) for i in 1:p)