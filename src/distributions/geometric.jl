export Geometric

import Distributions: Geometric, succprob, failprob

vague(::Type{<:Geometric}) = Geometric(Float64(tiny))

probvec(dist::Geometric) = (failprob(dist), succprob(dist))

prod_closed_rule(::Type{<:Geometric}, ::Type{<:Geometric}) = ClosedProd()

Base.prod(::ClosedProd, left::Geometric, right::Geometric) =
    Geometric(succprob(left) + succprob(right) - succprob(left) * succprob(right))

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Geometric) =
    KnownExponentialFamilyDistribution(Geometric, [log(one(Float64) - succprob(dist))])

Base.convert(::Type{Distribution}, η::KnownExponentialFamilyDistribution{Geometric}) =
    Geometric(one(Float64) - exp(first(getnaturalparameters(η))))

logpartition(η::KnownExponentialFamilyDistribution{Geometric}) = -log(one(Float64) - exp(first(getnaturalparameters(η))))

check_valid_natural(::Type{<:Geometric}, params) = length(params) == 1

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Geometric})
    η = first(getnaturalparameters(exponentialfamily))
    return (η <= zero(η)) && (η >= log(convert(typeof(η), tiny)))
end

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Geometric}, <:Geometric}, x) = 1.0
