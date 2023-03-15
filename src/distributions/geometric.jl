export Geometric

import Distributions: Geometric, succprob, failprob

vague(::Type{<:Geometric}) = Geometric(1e-12)

probvec(dist::Geometric) = (failprob(dist), succprob(dist))

#write analytical rule for product
prod_analytical_rule(::Type{<:Geometric}, ::Type{<:Geometric}) = ProdAnalyticalRuleAvailable()

Base.prod(::ProdAnalytical, left::Geometric, right::Geometric) =
    Geometric(succprob(left) + succprob(right) - succprob(left) * succprob(right))

# Geometric natural parameters 
Base.convert(::Type{NaturalParameters}, dist::Geometric) = NaturalParameters(Geometric, [log(1 - succprob(dist))])

Base.convert(::Type{Distribution}, η::NaturalParameters{Geometric}) = Geometric(1 - exp(first(get_params(η))))

lognormalizer(η::NaturalParameters{Geometric}) = -log(1 - exp(first(get_params(η))))

check_valid_natural(::Type{<:Geometric}, params) = length(params) == 1

function isproper(params::NaturalParameters{Geometric})
    η = first(get_params(params))
    return (η <= 0.0) && (η >= log(1e-12))
end

basemeasure(::Union{<:NaturalParameters{Geometric}, <:Geometric}, x) = 1.0
plus(::NaturalParameters{Geometric}, ::NaturalParameters{Geometric}) = Plus()