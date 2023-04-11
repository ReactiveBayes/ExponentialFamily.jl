export binomial_prod

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