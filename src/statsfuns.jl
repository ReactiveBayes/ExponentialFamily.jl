using StatsFuns: logistic
using StatsFuns: softmax, softmax!
using SpecialFunctions: gamma, loggamma

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