export NegativeBinomial
import Distributions: NegativeBinomial, probs
import StatsFuns: logit, logistic
import LogExpFunctions: logsumexp
import DomainSets: NaturalNumbers
using StaticArrays

BayesBase.vague(::Type{<:NegativeBinomial}, trials::Int) = NegativeBinomial(trials)
BayesBase.probvec(dist::NegativeBinomial) = (failprob(dist), succprob(dist))

Distributions.support(::Type{NegativeBinomial}) = NaturalNumbers()

function BayesBase.convert_paramfloattype(::Type{T}, distribution::NegativeBinomial) where {T <: Real}
    n, p = params(distribution)
    return NegativeBinomial(n, convert(AbstractVector{T}, p))
end

# NOTE: The product of two NegativeBinomial distributions is NOT a NegativeBinomial distribution.
function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: NegativeBinomial}
    rleft, rright = Integer(getconditioner(left)), Integer(getconditioner(right))

    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right

    sufficientstatistics = (identity,)

    function basemeasure(x)
        p_left, p_right, p_x = promote(rleft, rright, x)
        binomial_prod(p_x + p_left - 1, p_x + p_right - 1, p_x)
    end

    # The NB×NB product has infinite support (x = 0, 1, 2, …), so the normalizer is an
    # infinite sum. It is a terminating hypergeometric series only for special `r`; in
    # general we sum until the terms become negligible. The summand is unimodal in `x`
    # (log-binomials grow, `exp(η₁ x)` decays with η₁ < 0), so we accumulate in log-space
    # and stop once a term falls a relative tolerance below the running peak.
    function logpartition(η)
        η1 = η[1]
        logtol = log(1e-12)
        lterms = Float64[]
        lmax = -Inf
        x = 0
        while true
            lt = lchoose(x + rleft - 1, x) + lchoose(x + rright - 1, x) + η1 * x
            push!(lterms, lt)
            lmax = max(lmax, lt)
            (x > 0 && lt < lmax + logtol) && break
            x += 1
            x > 1_000_000 && break # safety cap against a non-convergent (η₁ ≥ 0) input
        end
        return logsumexp(lterms)
    end

    supp = NaturalNumbers()
    attributes = ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, supp)
    return ExponentialFamilyDistribution(
        Univariate,
        naturalparameters,
        nothing,
        attributes
    )
end

# Natural parametrization

function isproper(::NaturalParametersSpace, ::Type{NegativeBinomial}, η, conditioner::Number)
    if isnan(conditioner) || isinf(conditioner) || length(η) !== 1 || conditioner < 0
        return false
    end

    (η₁,) = unpack_parameters(NegativeBinomial, η)

    return !isnan(η₁) && !isinf(η₁) && η₁ <= 0
end

function isproper(::DefaultParametersSpace, ::Type{NegativeBinomial}, θ, conditioner::Number)
    if isnan(conditioner) || isinf(conditioner) || length(θ) !== 1 || conditioner < 0
        return false
    end

    (p,) = unpack_parameters(NegativeBinomial, θ)

    return !isnan(p) && !isinf(p) && (1 => p > 0)
end

function separate_conditioner(::Type{NegativeBinomial}, params)
    r, p = params
    return ((p,), r)
end

function join_conditioner(::Type{NegativeBinomial}, cparams, conditioner)
    (p,) = cparams
    r = conditioner
    return (r, p)
end

function (::MeanToNatural{NegativeBinomial})(tuple_of_θ::Tuple{Any}, _)
    (p,) = tuple_of_θ

    return (log(one(p) - p),)
end

function (::NaturalToMean{NegativeBinomial})(tuple_of_η::Tuple{Any}, _)
    (η₁,) = tuple_of_η
    return (one(η₁) - exp(η₁),)
end

function unpack_parameters(::Type{NegativeBinomial}, packed, _)
    return (first(packed),)
end

function unpack_parameters(::Type{NegativeBinomial}, packed)
    return (first(packed),)
end

isbasemeasureconstant(::Type{NegativeBinomial}) = NonConstantBaseMeasure()

getbasemeasure(::Type{NegativeBinomial}, conditioner) = (x) -> binomial(Int(x + conditioner - 1), x)

lchoose(a, b) = loggamma(a + 1) - loggamma(b + 1) - loggamma(a - b + 1)

getlogbasemeasure(::Type{NegativeBinomial}, conditioner) = (x) -> lchoose(Int(x + conditioner - 1), x)
getsufficientstatistics(::Type{NegativeBinomial}, conditioner) = (identity,)

getlogpartition(::NaturalParametersSpace, ::Type{NegativeBinomial}, conditioner) = (η) -> begin
    (η1,) = unpack_parameters(NegativeBinomial, η)
    return -conditioner * log(one(η1) - exp(η1))
end

getgradlogpartition(::NaturalParametersSpace, ::Type{NegativeBinomial}, conditioner) = (η) -> begin
    (η1,) = unpack_parameters(NegativeBinomial, η)
    return SA[-conditioner * (-exp(η1) / (one(η1) - exp(η1)));]
end

getfisherinformation(::NaturalParametersSpace, ::Type{NegativeBinomial}, r) = (η) -> begin
    (η1,) = unpack_parameters(NegativeBinomial, η)
    return SA[r * exp(η1) / (one(η1) - exp(η1))^2;;]
end

# Mean parametrization

getlogpartition(::DefaultParametersSpace, ::Type{NegativeBinomial}, conditioner) = (θ) -> begin
    (p,) = unpack_parameters(NegativeBinomial, θ)
    return -conditioner * log(one(p) - p)
end

getgradlogpartition(::DefaultParametersSpace, ::Type{NegativeBinomial}, conditioner) = (θ) -> begin
    (p,) = unpack_parameters(NegativeBinomial, η)
    return SA[conditioner * inv(one(p) - p);]
end

getfisherinformation(::DefaultParametersSpace, ::Type{NegativeBinomial}, r) = (θ) -> begin
    (p,) = unpack_parameters(NegativeBinomial, θ)
    return SA[r / (p^2 * (one(p) - p));;]
end
