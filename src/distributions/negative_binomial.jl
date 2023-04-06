export NegativeBinomial
import Distributions: NegativeBinomial, probs
import StatsFuns: logit, logistic
import DomainSets: NaturalNumbers

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

function Base.prod(::ClosedProd, left::NegativeBinomial, right::NegativeBinomial)
    rleft, _ = params(left)
    rright, _ = params(right)

    η_left = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, left)))
    η_right = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, right)))

    naturalparameters = [η_left + η_right]

    function basemeasure(x)
        p_left, p_right, p_x = promote(rleft, rright, x)
        b1 = binomial(p_x + p_left -1, p_x)
        b2 = binomial(p_x + p_right -1, p_x)
        result, flag = Base.mul_with_overflow(b1, b2)
        flag && return basemeasure(BigInt(left_trials), BigInt(right_trials), BigInt(x))
        return result
    end
    function logpartition_(η)
        max_m_n = max(m, n)
        exp_η = exp(η)
        max_m_n_plus1 = max_m_n + 1
        max_m_n_plus2 = max_m_n + 2

        term1 = _₂F₁(m, n, 1, exp_η)

        binomial1 = binomial(m + max_m_n, max_m_n_plus1)
        binomial2 = binomial(n + max_m_n, max_m_n_plus1)

        term2 = exp(η*(maximum([m, n])+1))

        term3 = binomial1*binomial2*_₃F₂(
            1,
            m + max_m_n_plus1,
            n + max_m_n_plus1,
            max_m_n_plus2,
            max_m_n_plus2,
            exp_η
        )

        result = log(term1 - term2 * term3)

        return result
    end
    
    supp = NaturalNumbers()

    return ExponentialFamilyDistribution(
        Float64,
        basemeasure,
        sufficientstatistics,
        naturalparameters,
        logpartition,
        supp
    )
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
    Base.isgreater(first(getnaturalparameters(exponentialfamily)), 0)

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial}) =
    -getconditioner(exponentialfamily) * log(one(Float64) - exp(first(getnaturalparameters(exponentialfamily))))

basemeasure(exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial}, x) =
    typeof(x) <: Integer ? binomial(x + getconditioner(exponentialfamily) - 1, x) : error("x must be integer")

function basemeasure(d::NegativeBinomial, x)
    @assert typeof(x) <: Integer "x must be integer"
    r, _ = params(d)
    return binomial(Int(x + r - 1), x)
end
