export Weibull

import Distributions: Weibull, Distribution

check_valid_natural(::Type{<:Weibull}, params) = length(params) === 1
check_valid_conditioner(::Type{<:Weibull}, conditioner) = isreal(conditioner) && conditioner > 0

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    η = getnaturalparameters(exponentialfamily)
    return first(η) < 0
end

basemeasure(dist::Weibull, x) = x^(shape(dist) - 1)
basemeasure(weibull::KnownExponentialFamilyDistribution{Weibull}, x) = x^(getconditioner(weibull) - 1)

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Weibull) =
    KnownExponentialFamilyDistribution(Weibull, [-(1 / scale(dist))^(shape(dist))], shape(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    k = get_condioner(exponentialfamily)
    η = first(get_params(exponentialfamily))
    return Weibull(k, (-1 / η)^(1 / k))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    return -log(-first(getnaturalparameters(exponentialfamily))) - log(getconditioner(exponentialfamily))
end
