export Geometric

import Distributions: Geometric, succprob, failprob

vague(::Type{<:Geometric}) = Geometric(1e-12)

probvec(dist::Geometric) = (failprob(dist), succprob(dist))

prod_analytical_rule(::Type{<:Geometric}, ::Type{<:Geometric}) = ProdAnalyticalRuleAvailable()

Base.prod(::ProdAnalytical, left::Geometric, right::Geometric) =
    Geometric(succprob(left) + succprob(right) - succprob(left) * succprob(right))

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Geometric) = ExponentialFamilyDistribution(Geometric, [log(1 - succprob(dist))])

Base.convert(::Type{Distribution}, η::ExponentialFamilyDistribution{Geometric}) = Geometric(1 - exp(first(getnaturalparameters(η))))

lognormalizer(η::ExponentialFamilyDistribution{Geometric}) = -log(1 - exp(first(getnaturalparameters(η))))

check_valid_natural(::Type{<:Geometric}, params) = length(params) == 1

function isproper(exponentialfamily::ExponentialFamilyDistribution{Geometric})
    η = first(getnaturalparameters(exponentialfamily))
    return (η <= 0.0) && (η >= log(1e-12))
end

basemeasure(::Union{<:ExponentialFamilyDistribution{Geometric}, <:Geometric}, x) = 1.0
plus(::ExponentialFamilyDistribution{Geometric}, ::ExponentialFamilyDistribution{Geometric}) = Plus()
