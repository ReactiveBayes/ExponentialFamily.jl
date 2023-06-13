export Binomial
using DomainSets
import Distributions: Binomial, probs
import StatsFuns: logit, logistic
import HypergeometricFunctions: _₂F₁

vague(::Type{<:Binomial}, trials::Int) = Binomial(trials)

probvec(dist::Binomial) = (failprob(dist), succprob(dist))

function convert_eltype(::Type{Binomial}, ::Type{T}, distribution::Binomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Binomial(n, convert(AbstractVector{T}, p))
end

prod_closed_rule(::Type{<:Binomial}, ::Type{<:Binomial}) = ClosedProd()

function Base.prod(::ClosedProd, left::Binomial, right::Binomial)
    efleft = convert(KnownExponentialFamilyDistribution, left)
    efright = convert(KnownExponentialFamilyDistribution, right)

    return prod(ClosedProd(), efleft, efright)
end

function Base.prod(
    ::ClosedProd,
    left::KnownExponentialFamilyDistribution{T},
    right::KnownExponentialFamilyDistribution{T}
) where {T <: Binomial}
    left_trials, right_trials = getconditioner(left), getconditioner(right)

    η_left = first(getnaturalparameters(left))
    η_right = first(getnaturalparameters(right))

    naturalparameters = [η_left + η_right]

    function basemeasure(x)
        p_left, p_right, p_x = promote(left_trials, right_trials, x)
        binomial_prod(p_left, p_right, p_x)
    end

    sufficientstatistics = (x) -> x
    logpartition = (η) -> log(_₂F₁(-left_trials, -right_trials, 1, exp(η)))
    supp = 0:max(left_trials, right_trials)

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

function fisherinformation(dist::Binomial)
    n, p = params(dist)
    return n / (p * (1 - p))
end

function fisherinformation(ef::KnownExponentialFamilyDistribution{Binomial})
    η = first(getnaturalparameters(ef))
    eη = exp(η)
    aux = eη / (1 + eη)
    n = getconditioner(ef)

    return n * aux * (1 - aux)
end
