export Geometric, GeometricNaturalParameters

import Distributions: Geometric, Distribution, succprob, failprob, logpdf

Distributions.cov(dist::Geometric) = var(dist)

vague(::Type{<:Geometric}) = Geometric(tiny)

probvec(dist::Geometric) = (failprob(dist), succprob(dist))

#write analytical rule for product
prod_analytical_rule(::Type{<:Geometric}, ::Type{<:Geometric}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Geometric, right::Geometric)
    return Geometric(succprob(left) + succprob(right) - succprob(left) * succprob(right)) 
end

# Geometric natural parameters 
struct GeometricNaturalParameters <: NaturalParameters
    η :: Real 
end

naturalparams(dist::Geometric) = GeometricNaturalParameters(log(1-succprob(dist) + 1e-12))

lognormalizer(params::GeometricNaturalParameters) = -log(1 - exp(params.η))

Distributions.logpdf(params::GeometricNaturalParameters,x) = x * params.η - lognormalizer(params.η)

Base.convert(::Type{Distribution}, params::GeometricNaturalParameters) = Geometric(1 - exp(params.η))
Base.:+(left::GeometricNaturalParameters, right::GeometricNaturalParameters) = GeometricNaturalParameters(left.η + right.η)
Base.:-(left::GeometricNaturalParameters, right::GeometricNaturalParameters) = GeometricNaturalParameters(left.η - right.η)

function Base.:(==)(left::GeometricNaturalParameters, right::GeometricNaturalParameters)
    return left.η == right.η
end

function isproper(params::GeometricNaturalParameters)
    return params.η <= 0
end
