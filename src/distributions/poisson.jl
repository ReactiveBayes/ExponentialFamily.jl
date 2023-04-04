export Poisson

import Distributions: Poisson, shape, scale, cov

Distributions.cov(dist::Poisson) = var(dist)

vague(::Type{<:Poisson}) = Poisson(1, huge)

prod_closed_rule(::Type{<:Poisson}, ::Type{<:Poisson}) = ClosedProd()

function Base.prod(::ClosedProd, left::Poisson, right::Poisson)
    return Poisson(rate(left)*rate(right))
end

function logpdf_sample_friendly(dist::Poisson)
    λ = params(dist)
    friendly = Poisson(λ)
    return (friendly, friendly)
end

check_valid_natural(::Type{<:Poisson}, params) = isequal(length(params), 1)

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Poisson) =
    KnownExponentialFamilyDistribution(Poisson, [log(rate(dist))])

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Poisson})
    η = first(getnaturalparameters(exponentialfamily))
    return Poisson(exp(η))
end

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Poisson}) = first(getnaturalparameters(exponentialfamily))

isproper(exponentialfamily::KnownExponentialFamilyDistribution{Poisson}) = Base.isgreater(first(getnaturalparameters(exponentialfamily)), 1)

basemeasure(::Union{<:KnownExponentialFamilyDistribution{Poisson}, <:Poisson}, x) = 1.0 / x
