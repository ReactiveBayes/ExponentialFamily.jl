export NegativeBinomial
import Distributions: NegativeBinomial, probs
import StatsFuns: logit, logistic
import DomainSets: NaturalNumbers
import HypergeometricFunctions: _₂F₁, _₃F₂

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
    rleft, rright = Integer(first(params(left))), Integer(first(params(right)))

    η_left = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, left)))
    η_right = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, right)))

    naturalparameters = [η_left + η_right]

    sufficientstatistics = (x) -> x

    function basemeasure(x)
        p_left, p_right, p_x = promote(rleft, rright, x)
        binomial_prod(p_x + p_left - 1, p_x + p_right - 1, p_x)
    end

    function logpartition(η)
        m, n = rright, rleft
        max_m_n = max(m, n)
        exp_η = exp(η)
        max_m_n_plus1 = max_m_n + 1
        max_m_n_plus2 = max_m_n + 2

        term1 = _₂F₁(m, n, 1, exp_η)

        term2 = exp(η * (maximum([m, n]) + 1))

        term3 =
            binomial_prod(m + max_m_n, n + max_m_n, max_m_n_plus1) * _₃F₂(
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

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial})
    η = first(getnaturalparameters(exponentialfamily))
    return Base.isequal(η, 0) || Base.isless(η, 0)
end
   
logpartition(exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial}) =
    -getconditioner(exponentialfamily) * log(one(Float64) - exp(first(getnaturalparameters(exponentialfamily))))

basemeasure(exponentialfamily::KnownExponentialFamilyDistribution{NegativeBinomial}, x) =
    typeof(x) <: Integer ? binomial(x + getconditioner(exponentialfamily) - 1, x) : error("x must be integer")

function basemeasure(d::NegativeBinomial, x)
    @assert typeof(x) <: Integer "x must be integer"
    r, _ = params(d)
    return binomial(Int(x + r - 1), x)
end
