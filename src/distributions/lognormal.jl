export LogNormal

import SpecialFunctions: digamma
import Distributions: LogNormal

Distributions.cov(dist::LogNormal) = var(dist)

vague(::Type{<:LogNormal}) = LogNormal(1, 1e12)

prod_closed_rule(::Type{<:LogNormal}, ::Type{<:LogNormal}) = ClosedProd()

function Base.prod(::ClosedProd, left::LogNormal, right::LogNormal)
    mean1, scale1 = params(left)
    mean2, scale2 = params(right)
    var1 = scale1^2
    var2 = scale2^2

    var3 = (var1 * var2) / (var1 + var2)
    mean3 = var3 * (mean1 / var1 + mean2 / var2 - 1)

    return LogNormal(mean3, sqrt(var3))
end

check_valid_natural(::Type{<:LogNormal}, params) = length(params) === 2

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::LogNormal)
    μ, scale = params(dist)
    var = scale^2
    return KnownExponentialFamilyDistribution(LogNormal, [(μ - var) / var, -1 / (2 * var)])
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{LogNormal})
    η = getnaturalparameters(exponentialfamily)
    η1 = first(η)
    η2 = getindex(η, 2)
    return LogNormal(-(η1 + 1) / (2 * η2), sqrt(-1 / (2 * η2)))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{LogNormal})
    η = getnaturalparameters(exponentialfamily)
    η1 = first(η)
    η2 = getindex(η, 2)
    return -(η1 + 1)^2 / (4 * η2)
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{LogNormal})
    η = getnaturalparameters(exponentialfamily)
    η2 = getindex(η, 2)
    return (η2 < 0)
end

function basemeasure(d::KnownExponentialFamilyDistribution{LogNormal}, x)
    η = getnaturalparameters(d)
    η2 = getindex(η, 2)
    return sqrt(-η2 / pi)
end

function basemeasure(d::LogNormal, x)
    var = varlogx(d)
    return 1 / (sqrt(2pi * var))
end

