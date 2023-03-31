export NegativeBinomial
import Distributions: NegativeBinomial, probs
import StatsFuns: logit, logistic

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

prod_closed_rule(::Type{<:NegativeBinomial}, ::Type{<:NegativeBinomial}) = ClosedProd()

function Base.prod(::ClosedProd, left::NegativeBinomial, right::NegativeBinomial)
    rleft, left_p = params(left)
    rright, right_p = params(right)
    @assert rleft == rright "Number of trials in $(left) and $(right) is not equal"

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > zero(norm) "Product of $(left) and $(right) results in non-normalizable distribution"
    return NegativeBinomial(rleft, pprod / norm)
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::NegativeBinomial)
    n, p = params(dist)
    return KnownExponentialFamilyDistribution(NegativeBinomial, [log(p)], n)
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial})
    return NegativeBinomial(getconditioner(exponentialfamily), exp(first(getnaturalparameters(exponentialfamily))))
end

check_valid_natural(::Type{<:NegativeBinomial}, params) = length(params) == 1

function check_valid_conditioner(::Type{<:NegativeBinomial}, conditioner)
    isinteger(conditioner) && conditioner > zero(conditioner)
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial}) =
    getconditioner(exponentialfamily) > zero(Int64) ? true : false

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial}) =
    -getconditioner(exponentialfamily) * log(one(Float64) - exp(first(getnaturalparameters(exponentialfamily))))

basemeasure(exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial}, x) =
    typeof(x) <: Integer ? binomial(x + getconditioner(exponentialfamily) - 1, x) : error("x must be integer")

function basemeasure(d::NegativeBinomial, x)
    @assert typeof(x) <: Integer "x must be integer"
    r, _ = params(d)
    return binomial(Int(x + r - 1), x)
end
