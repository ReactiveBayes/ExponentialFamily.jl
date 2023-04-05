export Binomial
using DomainSets
import Distributions: Binomial, probs
import StatsFuns: logit, logistic
import HypergeometricFunctions: pFq

vague(::Type{<:Binomial}, trials::Int) = Binomial(trials)

probvec(dist::Binomial) = (failprob(dist), succprob(dist))

function convert_eltype(::Type{Binomial}, ::Type{T}, distribution::Binomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Binomial(n, convert(AbstractVector{T}, p))
end

prod_closed_rule(::Type{<:Binomial}, ::Type{<:Binomial}) = ClosedProd()

function Base.prod(::ClosedProd, left::Binomial, right::Binomial)
    left_trials, right_trials = ntrials(left), ntrials(right)

    η_left = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, left)))
    η_right = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, right)))

    naturalparameters = [η_left + η_right]

    function basemeasure(x)
        i_left = left_trials::BigInt > 40 ? BigInt(left_trials) : Int(left_trials)
        i_right = right_trials::BigInt > 40 ? BigInt(right_trials) : Int(right_trials)
        binomial(i_left, x) * binomial(i_right, x)
    end

    sufficientstatistics = (x) -> x
    logpartition = (η) -> log(pFq([-left_trials, -right_trials], [1], exp(η)))
    supp = support(left)

    return ExponentialFamilyDistribution(
        Float64,
        basemeasure,
        sufficientstatistics,
        naturalparameters,
        logpartition,
        supp
    )
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Binomial)
    n, p = params(dist)
    return KnownExponentialFamilyDistribution(Binomial, [logit(p)], n)
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Binomial})
    return Binomial(getconditioner(exponentialfamily), logistic(first(getnaturalparameters(exponentialfamily))))
end

check_valid_natural(::Type{<:Binomial}, params) = length(params) == 1

function check_valid_conditioner(::Type{<:Binomial}, conditioner)
    isinteger(conditioner) && conditioner > zero(conditioner)
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{Binomial}) =
    getconditioner(exponentialfamily) > zero(Int64) ? true : false

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Binomial}) =
    getconditioner(exponentialfamily)log(one(Float64) + exp(first(getnaturalparameters(exponentialfamily))))

basemeasure(exponentialfamily::KnownExponentialFamilyDistribution{Binomial}, x) =
    typeof(x) <: Integer ? binomial(getconditioner(exponentialfamily), x) : error("x must be integer")

function basemeasure(d::Binomial, x)
    binomial(d.n, x)
end
