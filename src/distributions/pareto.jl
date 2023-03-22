export Pareto

import Distributions: Pareto, shape, scale, params

vague(::Type{<:Pareto}) = Pareto(1e12)

Distributions.cov(dist::Type{<:Pareto}) = var(dist)

prod_analytical_rule(::Type{<:Pareto}, ::Type{<:Pareto}) = ConditionallyClosedProd()

# function Base.prod(::ClosedProd, left::L, right::R) where {L <: Pareto, R <: Pareto}
#     n1 = shape(left) + shape(right) + 1
#     n2 = exp(
#         shape(left) / (shape(left) + shape(right) + 1) * log(scale(left)) +
#         shape(right) / (shape(left) + shape(right) + 1) * log(scale(right))
#     )
#     return Pareto(n1, n2)
# end

function Distributions.mean(dist::Pareto)
    k, θ = params(dist)
    k > 1 ? k * θ / (k - 1) : Inf
end

function logpdf_sample_friendly(dist::Pareto)
    friendly = convert(Pareto, dist)
    return (friendly, friendly)
end

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Pareto) =
    KnownExponentialFamilyDistribution(Pareto, [-shape(dist) - 1], shape(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{<:Pareto})
    η = first(getnaturalparameters(exponentialfamily))
    return Pareto(-1 - η)
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

# function plus(np1::KnownExponentialFamilyDistribution{Pareto}, np2::KnownExponentialFamilyDistribution{Pareto})
#     if getconditioner(np1) == getconditioner(np2) && (first(size(getnaturalparameters(np1))) == first(size(getnaturalparameters(np2))))
#         return Plus()
#     else
#         return Concat()
#     end
# end
