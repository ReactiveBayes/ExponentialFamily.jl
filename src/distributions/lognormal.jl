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
    return KnownExponentialFamilyDistribution(LogNormal, [μ / var, -1 / (2 * var)])
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{LogNormal})
    η = getnaturalparameters(exponentialfamily)
    η1 = first(η)
    η2 = getindex(η, 2)
    return LogNormal(-η1 / (2 * η2), sqrt(-1 / (2 * η2)))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{LogNormal})
    η = getnaturalparameters(exponentialfamily)
    η1 = first(η)
    η2 = getindex(η, 2)
    return -(η1)^2 / (4 * η2) - log(-2η2) / 2
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{LogNormal})
    η = getnaturalparameters(exponentialfamily)
    η2 = getindex(η, 2)
    return (η2 < 0)
end

support(::Union{<:KnownExponentialFamilyDistribution{LogNormal}, <:LogNormal}) = ClosedInterval{Real}(0, Inf)

function basemeasure(ef::KnownExponentialFamilyDistribution{LogNormal}, x::Real)
    @assert insupport(ef, x) "Lognormal should be evaluated at positive values"
    return 1 / (sqrt(2 * pi) * x)
end

function basemeasure(dist::LogNormal, x::Real)
    @assert insupport(dist, x) "Lognormal should be evaluated at positive values"
    return 1 / (sqrt(2 * pi) * x)
end

function fisherinformation(d::LogNormal)
    σ = d.σ
    return [1/(σ^2) 0.0; 0.0 2/(σ^2)]
end

function fisherinformation(ef::KnownExponentialFamilyDistribution{LogNormal})
    η = getnaturalparameters(ef)
    η1 = getindex(η, 1)
    η2 = getindex(η, 2)
    return [-1/(2η2) (η1)/(2η2^2); (η1)/(2η2^2) -(η1)^2/(2*(η2^3))+1/(2*η2^2)]
end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{LogNormal}, x::Real)
    @assert insupport(ef, x) "Lognormal should be evaluated at positive values"
    return [log(x), log(x)^2]
end
