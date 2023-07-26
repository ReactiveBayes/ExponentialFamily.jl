export Geometric

import Distributions: Geometric, succprob, failprob
using StaticArrays

vague(::Type{<:Geometric}) = Geometric(Float64(tiny))

probvec(dist::Geometric) = (failprob(dist), succprob(dist))

closed_prod_rule(::Type{<:Geometric}, ::Type{<:Geometric}) = ClosedProd()

Base.prod(::ClosedProd, left::Geometric, right::Geometric) =
    Geometric(succprob(left) + succprob(right) - succprob(left) * succprob(right))

pack_naturalparameters(dist::Geometric) = [log(one(Float64) - succprob(dist))]
function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Geometric}) 
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]
    return η1
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Geometric) =
    ExponentialFamilyDistribution(Geometric, pack_naturalparameters(dist))

Base.convert(::Type{Distribution}, η::ExponentialFamilyDistribution{Geometric}) =
    Geometric(one(Float64) - exp(unpack_naturalparameters(η)))

logpartition(η::ExponentialFamilyDistribution{Geometric}) =
    -log(one(Float64) - exp(unpack_naturalparameters(η)))

check_valid_natural(::Type{<:Geometric}, params) = length(params) == 1

function isproper(exponentialfamily::ExponentialFamilyDistribution{Geometric})
    η = unpack_naturalparameters(exponentialfamily)
    return (η <= zero(η)) && (η >= log(convert(typeof(η), tiny)))
end

function insupport(::ExponentialFamilyDistribution{Geometric, P, C, Safe}, x::Real) where {P, C}
    return zero(Float64) < x && x < Inf && typeof(x) <: Int
end

function basemeasure(union::Union{<:ExponentialFamilyDistribution{Geometric}, <:Geometric}, x::Real)
    @assert insupport(union, x) "$(x) is not in the support of Geometric distribution"
    return one(x)
end
function fisherinformation(exponentialfamily::ExponentialFamilyDistribution{Geometric})
    η = unpack_naturalparameters(exponentialfamily)
    SA[exp(η) / (one(Float64) - exp(η))^2]
end

function fisherinformation(dist::Geometric)
    p = succprob(dist)
    SA[one(Float64) / (p * (one(Float64) - p)) + one(Float64) / p^2]
end

function sufficientstatistics(union::Union{<:ExponentialFamilyDistribution{Geometric}, <:Geometric}, x::Real)
    @assert insupport(union, x) "$(x) is not in the support of Geometric distribution"
    return SA[x]
end
