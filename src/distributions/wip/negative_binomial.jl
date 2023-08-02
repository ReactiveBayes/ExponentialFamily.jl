export NegativeBinomial
import Distributions: NegativeBinomial, probs
import StatsFuns: logit, logistic
import DomainSets: NaturalNumbers
using StaticArrays

vague(::Type{<:NegativeBinomial}, trials::Int) = NegativeBinomial(trials)

probvec(dist::NegativeBinomial) = (failprob(dist), succprob(dist))

function convert_eltype(
    ::Type{NegativeBinomial},
    ::Type{T},
    distribution::NegativeBinomial{R}
) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return NegativeBinomial(n, convert(AbstractVector{T}, p))
end

closed_prod_rule(::Type{<:NegativeBinomial}, ::Type{<:NegativeBinomial}) = ClosedProd()

# NOTE: The product of two NegativeBinomial distributions is NOT a NegativeBinomial distribution.
function Base.prod(
    ::ClosedProd,
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: NegativeBinomial}
    rleft, rright = Integer(getconditioner(left)), Integer(getconditioner(right))

    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right

    sufficientstatistics = (x) -> SA[x]

    function basemeasure(x)
        p_left, p_right, p_x = promote(rleft, rright, x)
        binomial_prod(p_x + p_left - 1, p_x + p_right - 1, p_x)
    end

    function logpartition(η)
        return log(sum(binomial_prod(x + rleft - 1, x + rright - 1, x) * exp(η[1] * x) for x in 0:max(rright, rleft)))
    end

    supp = NaturalNumbers()

    return ExponentialFamilyDistribution(
        Univariate,
        naturalparameters,
        nothing,
        basemeasure,
        sufficientstatistics,
        logpartition,
        supp
    )
end

function Base.prod(::ClosedProd, left::T, right::T) where {T <: NegativeBinomial}
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)

    return prod(ef_left, ef_right)
end

pack_naturalparameters(dist::NegativeBinomial) = [log(one(Float64) - params(dist)[2])]
function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:NegativeBinomial})
    η = getnaturalparameters(ef)
    @inbounds η1 = η[1]

    return (η1,)
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::NegativeBinomial)
    n, _ = params(dist)
    return ExponentialFamilyDistribution(NegativeBinomial, pack_naturalparameters(dist), n)
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{NegativeBinomial})
    return NegativeBinomial(
        getconditioner(exponentialfamily),
        one(Float64) - exp(first(unpack_naturalparameters(exponentialfamily)))
    )
end

check_valid_natural(::Type{<:NegativeBinomial}, params) = length(params) == 1

function check_valid_conditioner(::Type{<:NegativeBinomial}, conditioner)
    isinteger(conditioner) && conditioner > zero(conditioner)
end

function isproper(exponentialfamily::ExponentialFamilyDistribution{NegativeBinomial})
    (η,) = unpack_naturalparameters(exponentialfamily)
    return η ≤ 0
end

logpartition(exponentialfamily::ExponentialFamilyDistribution{NegativeBinomial}) =
    -getconditioner(exponentialfamily) * log(one(Float64) - exp(first(unpack_naturalparameters(exponentialfamily))))

support(::ExponentialFamilyDistribution{NegativeBinomial}) = NaturalNumbers()

basemeasureconstant(::ExponentialFamilyDistribution{NegativeBinomial}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:NegativeBinomial}) = NonConstantBaseMeasure()
basemeasure(ef::ExponentialFamilyDistribution{NegativeBinomial}) = (x) -> basemeasure(ef, x)
function basemeasure(exponentialfamily::ExponentialFamilyDistribution{NegativeBinomial}, x::Real)
    return binomial(Int(x + getconditioner(exponentialfamily) - 1), x)
end
function basemeasure(d::NegativeBinomial, x::Real)
    r, _ = params(d)
    return binomial(Int(x + r - 1), x)
end

function fisherinformation(ef::ExponentialFamilyDistribution{NegativeBinomial})
    r = getconditioner(ef)
    (η,) = unpack_naturalparameters(ef)
    return SA[r * exp(η) / (one(Float64) - exp(η))^2;;]
end

function fisherinformation(dist::NegativeBinomial)
    r, p = params(dist)
    SA[r / (p^2 * (one(p) - p));;]
end

sufficientstatistics(ef::ExponentialFamilyDistribution{NegativeBinomial}) = (x) -> sufficientstatistics(ef, x)
function sufficientstatistics(
    ::ExponentialFamilyDistribution{NegativeBinomial},
    x::Real
)
    return SA[x]
end
