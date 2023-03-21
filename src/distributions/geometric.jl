export Geometric

import Distributions: Geometric, succprob, failprob

vague(::Type{<:Geometric}) = Geometric(1e-12)

probvec(dist::Geometric) = (failprob(dist), succprob(dist))

prod_analytical_rule(::Type{<:Geometric}, ::Type{<:Geometric}) = ClosedProd()

Base.prod(::ClosedProd, left::Geometric, right::Geometric) =
    Geometric(succprob(left) + succprob(right) - succprob(left) * succprob(right))

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Geometric) = KnownExponentialFamilyDistribution(Geometric, [log(1 - succprob(dist))])

Base.convert(::Type{Distribution}, η::KnownExponentialFamilyDistribution{Geometric}) = Geometric(1 - exp(first(getnaturalparameters(η))))

logpartition(η::KnownExponentialFamilyDistribution{Geometric}) = -log(1 - exp(first(getnaturalparameters(η))))

check_valid_natural(::Type{<:Geometric}, params) = length(params) == 1

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Geometric})
    η = first(getnaturalparameters(exponentialfamily))
    return (η <= 0.0) && (η >= log(1e-12))
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Geometric}, <:Geometric}, x) = 1.0
