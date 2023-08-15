export Binomial

using DomainSets

import Distributions: Binomial, probs, Univariate
import StatsFuns: logit, logistic, log1pexp

import HypergeometricFunctions: _₂F₁
import StaticArrays: SA

vague(::Type{<:Binomial}, trials::Int) = Binomial(trials)

probvec(dist::Binomial) = (failprob(dist), succprob(dist))

function convert_paramfloatype(::Type{Binomial}, ::Type{T}, distribution::Binomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Binomial(n, convert(T, p))
end

default_prod_rule(::Type{<:Binomial}, ::Type{<:Binomial}) = PreserveTypeProd(ExponentialFamilyDistribution)

function Base.prod(::PreserveTypeProd{ExponentialFamilyDistribution}, left::Binomial, right::Binomial)
    efleft = convert(ExponentialFamilyDistribution, left)
    efright = convert(ExponentialFamilyDistribution, right)
    return prod(PreserveTypeProd(ExponentialFamilyDistribution), efleft, efright)
end

# NOTE: The product of two Binomial distributions is NOT a Binomial distribution.
function Base.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Binomial}
    left_trials, right_trials = getconditioner(left), getconditioner(right)

    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right

    basemeasure = (x) -> binomial_prod(left_trials, right_trials, x)
    sufficientstatistics = (identity,)
    logpartition = (η) -> log(_₂F₁(-left_trials, -right_trials, 1, exp(first(η))))
    support = 0:max(left_trials, right_trials)

    attributes = ExponentialFamilyDistributionAttributes(
        basemeasure,
        sufficientstatistics,
        logpartition,
        support
    )

    return ExponentialFamilyDistribution(
        Univariate,
        naturalparameters,
        nothing,
        attributes
    )
end

# Natural parametrization

# This is needed, because `getsupport` of Binomial is not really defined with respect to its type due to need in ntrials
Distributions.insupport(ef::ExponentialFamilyDistribution{Binomial}, x) = insupport(convert(Distribution, ef), x)

isproper(::NaturalParametersSpace, ::Type{Binomial}, η, conditioner) = length(η) === 1 && isinteger(conditioner) && conditioner >= 0

isproper(::MeanParametersSpace, ::Type{Binomial}, θ, conditioner::Number) = length(θ) === 1 && 0 <= first(θ) <= 1 && isinteger(conditioner) && conditioner >= 0

function separate_conditioner(::Type{Binomial}, params)
    ntrials, succprob = params
    return ((succprob,), ntrials)
end

function join_conditioner(::Type{Binomial}, cparams, conditioner)
    (succprob,) = cparams
    ntrials = conditioner
    return (ntrials, succprob)
end

function (::MeanToNatural{Binomial})(tuple_of_θ::Tuple{Any}, _)
    (succprob,) = tuple_of_θ
    return (logit(succprob),)
end

function (::NaturalToMean{Binomial})(tuple_of_η::Tuple{Any}, _)
    (η₁,) = tuple_of_η
    return (logistic(η₁),)
end

function unpack_parameters(::Type{Binomial}, packed)
    return (first(packed),)
end

isbasemeasureconstant(::Type{Binomial}) = NonConstantBaseMeasure()

getbasemeasure(::Type{Binomial}, ntrials) = Base.Fix1(binomial, ntrials)
getsufficientstatistics(::Type{Binomial}, _) = (identity,)

getlogpartition(::NaturalParametersSpace, ::Type{Binomial}, ntrials) = (η) -> begin
    (η₁,) = unpack_parameters(Binomial, η)
    return ntrials * log1pexp(η₁)
end

getfisherinformation(::NaturalParametersSpace, ::Type{Binomial}, ntrials) = (η) -> begin
    (η₁,) = unpack_parameters(Binomial, η)
    aux = logistic(η₁)
    return SA[ntrials * aux * (1 - aux);;]
end

## Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Binomial}, ntrials) = (θ) -> begin
    (p,) = unpack_parameters(Binomial, θ)
    return -ntrials * log(1 - p)
end

getfisherinformation(::MeanParametersSpace, ::Type{Binomial}, ntrials) = (θ) -> begin
    (p,) = unpack_parameters(Binomial, θ)
    return SA[ntrials / (p * (1 - p));;]
end