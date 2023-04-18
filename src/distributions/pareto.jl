export Pareto

import Distributions: Pareto, shape, scale, params

vague(::Type{<:Pareto}) = Pareto(1e12)

Distributions.cov(dist::Type{<:Pareto}) = var(dist)

prod_closed_rule(::Type{<:Pareto}, ::Type{<:Pareto}) = ClosedProd()

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Pareto) =
    KnownExponentialFamilyDistribution(Pareto, [-shape(dist) - 1], scale(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{<:Pareto})
    η = first(getnaturalparameters(exponentialfamily))
    conditioner = getconditioner(exponentialfamily)
    return Pareto(-1 - η, conditioner)
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Pareto})
    η = first(getnaturalparameters(exponentialfamily))
    k = getconditioner(exponentialfamily)
    return -log(-1 - η) + (1 + η)log(k)
end

check_valid_natural(::Type{<:Pareto}, params) = (length(params) === 1)
check_valid_conditioner(::Type{<:Pareto}, conditioner) = isinteger(conditioner) && conditioner > 0
function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Pareto})
    η = getnaturalparameters(exponentialfamily)
    return (first(η) <= -1)
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Pareto}, <:Pareto}, x) = 1.0
