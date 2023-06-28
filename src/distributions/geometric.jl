export Geometric

import Distributions: Geometric, succprob, failprob

vague(::Type{<:Geometric}) = Geometric(Float64(tiny))

probvec(dist::Geometric) = (failprob(dist), succprob(dist))

closed_prod_rule(::Type{<:Geometric}, ::Type{<:Geometric}) = ClosedProd()

Base.prod(::ClosedProd, left::Geometric, right::Geometric) =
    Geometric(succprob(left) + succprob(right) - succprob(left) * succprob(right))

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Geometric) =
    KnownExponentialFamilyDistribution(Geometric, log(one(Float64) - succprob(dist)))

Base.convert(::Type{Distribution}, η::KnownExponentialFamilyDistribution{Geometric}) =
    Geometric(one(Float64) - exp(getnaturalparameters(η)))

logpartition(η::KnownExponentialFamilyDistribution{Geometric}) =
    -log(one(Float64) - exp(getnaturalparameters(η)))

check_valid_natural(::Type{<:Geometric}, params) = length(params) == 1

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Geometric})
    η = getnaturalparameters(exponentialfamily)
    return (η <= zero(η)) && (η >= log(convert(typeof(η), tiny)))
end

function insupport(::KnownExponentialFamilyDistribution{Geometric, P, C, Safe}, x) where {P, C}
    return zero(Float64) < x && x < Inf && typeof(x) <: Int
end

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Geometric}, <:Geometric}, x::Real)
    @assert insupport(union, x) "$(x) is not in the support of Geometric distribution"
    return one(typeof(x))
end
function fisherinformation(exponentialfamily::KnownExponentialFamilyDistribution{Geometric})
    η = getnaturalparameters(exponentialfamily)
    exp(η) / (one(Float64) - exp(η))^2
end

function fisherinformation(dist::Geometric)
    p = succprob(dist)
    one(Float64) / (p * (one(Float64) - p)) + one(Float64) / p^2
end

function sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{Geometric}, <:Geometric}, x::Real)
    @assert insupport(union, x) "$(x) is not in the support of Geometric distribution"
    return x
end
