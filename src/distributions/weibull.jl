export Weibull

import Distributions: Weibull, Distribution

check_valid_natural(::Type{<:Weibull}, params) = length(params) === 1
check_valid_conditioner(::Type{<:Weibull}, conditioner) = isreal(conditioner) && conditioner > 0

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    η = get_params(exponentialfamily)
    return first(η) < 0
end

basemeasure(dist::Weibull, x) = x^(shape(dist) - 1)
basemeasure(weibull::KnownExponentialFamilyDistribution{Weibull}, x) = x^(get_conditioner(weibull) - 1)

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Weibull) =
    KnownExponentialFamilyDistribution(Weibull, [-(1 / scale(dist))^(shape(dist))], shape(dist))

function Base.convert(::Type{Distribution}, np::NaturalParameters{Weibull})
    k = get_condioner(np)
    η = first(get_params(np))
    return Weibull(k, (-1 / η)^(1 / k))
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Weibull})
    return -log(first(get_params(exponentialfamily))) - log(get_condioner(exponentialfamily))
end