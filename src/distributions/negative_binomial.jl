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
function Base.prod(
    ::ClosedProd,
    left::KnownExponentialFamilyDistribution{T},
    right::KnownExponentialFamilyDistribution{T}
) where {T <: NegativeBinomial}
    rleft, rright = Integer(first(getconditioner(left))), Integer(first(getconditioner(right)))

    η_left = first(getnaturalparameters(left))
    η_right = first(getnaturalparameters(right))

    naturalparameters = [η_left + η_right]

    sufficientstatistics = (x) -> x

    function basemeasure(x)
        p_left, p_right, p_x = promote(rleft, rright, x)
        binomial_prod(p_x + p_left - 1, p_x + p_right - 1, p_x)
    end

    function logpartition(η)
        return log(sum(binomial_prod(x + rleft - 1, x + rright - 1, x) * exp(η * x) for x in 0:max(rright, rleft)))
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

function Base.prod(::ClosedProd, left::NegativeBinomial, right::NegativeBinomial)
    ef_left = convert(KnownExponentialFamilyDistribution, left)
    ef_right = convert(KnownExponentialFamilyDistribution, right)

    return prod(ef_left, ef_right)
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
    return η ≤ 0
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

function fisherinformation(ef::KnownExponentialFamilyDistribution{NegativeBinomial})
    r = getconditioner(ef)
    η = first(getnaturalparameters(ef))
    return r * exp(η) / (one(Float64) - exp(η))^2
end

function fisherinformation(dist::NegativeBinomial)
    r, p = params(dist)
    r / (p^2 * (one(p) - p))
end


sufficientstatistics(::Union{<:KnownExponentialFamilyDistribution{NegativeBinomial}, <:NegativeBinomial}, x) = x