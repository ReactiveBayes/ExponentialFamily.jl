export Weibull

import Distributions: Weibull, Distribution

check_valid_natural(::Type{<:Weibull}, params) = length(params) === 1
check_valid_conditioner(::Type{<:Weibull}, conditioner) = isreal(conditioner) && conditioner > 0

function isproper(params::NaturalParameters{Weibull})
    η = get_params(params)
    return first(η) < 0
end

basemeasure(np::NaturalParameters{Weibull}, x) = x^(get_conditioner(np) - 1)
basemeasure(dist::Weibull, x) = x^(shape(dist) - 1)

Base.convert(::Type{NaturalParameters}, dist::Pareto) =
    NaturalParameters(Weibull, [-(1 / scale(dist))^(shape(dist))], shape(dist))

function Base.convert(::Type{Distribution}, np::NaturalParameters{Weibull})
    k = get_condioner(np)
    η = first(get_params(np))
    return Weibull(k, (-1 / η)^(1 / k))
end

function lognormalizer(np::NaturalParameters{Weibull})
    return -log(first(get_params(np))) - log(get_condioner(np))
end
