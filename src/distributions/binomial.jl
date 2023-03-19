export Binomial
import Distributions: Binomial, probs
import StatsFuns: logit, logistic

vague(::Type{<:Binomial}, trials::Int) = Binomial(trials)

probvec(dist::Binomial) = (failprob(dist), succprob(dist))

function convert_eltype(::Type{Binomial}, ::Type{T}, distribution::Binomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Binomial(n, convert(AbstractVector{T}, p))
end

prod_analytical_rule(::Type{<:Binomial}, ::Type{<:Binomial}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Binomial, right::Binomial)
    @assert ntrials(left) == ntrials(right) "Number of trials in $(left) and $(right) is not equal"
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Binomial(ntrials(left), pprod / norm)
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Binomial)
    n, p = params(dist)
    return ExponentialFamilyDistribution(Binomial, [logit(p)], n)
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Binomial})
    return Binomial(getconditioner(exponentialfamily), logistic(first(getnaturalparameters(exponentialfamily))))
end

check_valid_natural(::Type{<:Binomial}, params) = length(params) == 1

function check_valid_conditioner(::Type{<:Binomial}, conditioner)
    isinteger(conditioner) && conditioner > 0
end

isproper(exponentialfamily::ExponentialFamilyDistribution{Binomial}) = getconditioner(exponentialfamily) > 0 ? true : false

lognormalizer(exponentialfamily::ExponentialFamilyDistribution{Binomial}) = getconditioner(exponentialfamily)log(1 + exp(first(getnaturalparameters(exponentialfamily))))

basemeasure(exponentialfamily::ExponentialFamilyDistribution{Binomial}, x) =
    typeof(x) <: Integer ? binomial(getconditioner(exponentialfamily), x) : error("x must be integer")

function basemeasure(d::Binomial, x)
    binomial(d.n, x)
end

function plus(ef1::ExponentialFamilyDistribution{Binomial}, ef2::ExponentialFamilyDistribution{Binomial})
    condition = getconditioner(ef1) == getconditioner(ef2) && (length(getnaturalparameters(ef1)) == length(getnaturalparameters(ef2)))
    return condition ? Plus() : Concat()
end
