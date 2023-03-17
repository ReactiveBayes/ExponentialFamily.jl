export Pareto

import Distributions: Pareto, shape, scale, params

vague(::Type{<:Pareto}) = Pareto(1e12)

Distributions.cov(dist::Type{<:Pareto}) = var(dist)

prod_analytical_rule(::Type{<:Pareto}, ::Type{<:Pareto}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::L, right::R) where {L <: Pareto, R <: Pareto}
    n1 = shape(left) + shape(right) + 1
    n2 = exp(
        shape(left) / (shape(left) + shape(right) + 1) * log(scale(left)) +
        shape(right) / (shape(left) + shape(right) + 1) * log(scale(right))
    )
    return Pareto(n1, n2)
end

function logpdf_sample_friendly(dist::Pareto)
    friendly = convert(Pareto, dist)
    return (friendly, friendly)
end

Base.convert(::Type{NaturalParameters}, dist::Pareto) =
    NaturalParameters(Pareto, [-shape(dist) - 1], shape(dist))

function Base.convert(::Type{Distribution}, params::NaturalParameters{<:Pareto})
    η = first(get_params(params))
    return Pareto(-1 - η)
end

function lognormalizer(params::NaturalParameters{Pareto})
    η = first(get_params(params))
    k = get_conditioner(params)
    return -log(-1 - η) + (1 + η)log(k)
end

check_valid_natural(::Type{<:Pareto}, params) = length(params) === 1
check_valid_conditioner(::Type{<:Pareto}, conditioner) = isinteger(conditioner) && conditioner > 0
function isproper(params::NaturalParameters{Pareto})
    η = get_params(params)
    return (first(η) <= -1)
end

basemeasure(::Union{<:NaturalParameters{Pareto}, <:Pareto}, x) = 1.0

function plus(np1::NaturalParameters{Pareto}, np2::NaturalParameters{Pareto})
    if get_conditioner(np1) == get_conditioner(np2) && (first(size(get_params(np1))) == first(size(get_params(np2))))
        return Plus()
    else
        return Concat()
    end
end
