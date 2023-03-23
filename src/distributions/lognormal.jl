export LogNormal

import SpecialFunctions: digamma
import Distributions: LogNormal

Distributions.cov(dist::LogNormal) = var(dist)

vague(::Type{<:LogNormal}) = LogNormal(1, 1e12)

prod_analytical_rule(::Type{<:LogNormal}, ::Type{<:LogNormal}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::LogNormal, right::LogNormal)
    mean1, var1 = params(left)
    mean2, var2 = params(right)

    var3 = (var1 * var2) / (var1 + var2)
    mean3 = var3 * (mean1 / var1 + mean2 / var2 - 1)

    return LogNormal(mean3, var3)
end

check_valid_natural(::Type{<:LogNormal}, params) = length(params) === 2

function Base.convert(::Type{NaturalParameters}, dist::LogNormal)
    μ, var = params(dist)
    return NaturalParameters(LogNormal, [(μ - var) / var, -1 / (2 * var)])
end

function Base.convert(::Type{Distribution}, params::NaturalParameters{LogNormal})
    η = get_params(params)
    η1 = first(η)
    η2 = getindex(η, 2)
    return LogNormal(-(η1 + 1) / (2 * η2), -1 / (2 * η2))
end

function lognormalizer(params::NaturalParameters{LogNormal})
    η = get_params(params)
    η1 = first(η)
    η2 = getindex(η, 2)
    return -(η1 + 1)^2 / (4 * η2)
end

function isproper(params::NaturalParameters{LogNormal})
    η = get_params(params)
    η2 = getindex(η, 2)
    return (η2 < 0)
end

function basemeasure(d::Union{<:NaturalParameters{LogNormal}, <:LogNormal}, x)
    if typeof(d) <: LogNormal
        μ, var = params(d)

    else
        η = get_params(d)
        η2 = getindex(η, 2)
        var = -1 / (2 * η2)
    end
    return 1 / (sqrt(2pi * var))
end
plus(::NaturalParameters{LogNormal}, ::NaturalParameters{LogNormal}) = Plus()
