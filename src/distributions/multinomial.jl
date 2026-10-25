export Multinomial

import Distributions: Multinomial, probs
import DomainSets: Domain
import LogExpFunctions: logaddexp, logsumexp, softmax, softmax!

BayesBase.vague(::Type{<:Multinomial}, n::Int, dims::Int) = Multinomial(n, dims)
BayesBase.probvec(dist::Multinomial) = probs(dist)
BayesBase.convert_paramfloattype(::Type{T}, distribution::Multinomial) where {T <: Real} =
    Multinomial(distribution.n, convert(AbstractVector{T}, probvec(distribution)))

BayesBase.default_prod_rule(::Type{<:Multinomial}, ::Type{<:Multinomial}) = PreserveTypeProd(ExponentialFamilyDistribution)

# Both the ordinary family and its product live on count vectors, not on 0:n.
struct _MultinomialSupport <: Domain{AbstractVector}
    trials::Int
    categories::Int
end

Base.eltype(::_MultinomialSupport) = AbstractVector
BayesBase.value_support(::Type{_MultinomialSupport}) = Discrete
Base.in(x, ::_MultinomialSupport) = false
function Base.in(x::AbstractVector{<:Real}, support::_MultinomialSupport)
    length(x) == support.categories || return false
    remaining = support.trials
    for xi in x
        (isinteger(xi) && 0 <= xi <= remaining) || return false
        remaining -= xi
    end
    return iszero(remaining)
end

getsupport(ef::ExponentialFamilyDistribution{Multinomial}) = _MultinomialSupport(getconditioner(ef), length(getnaturalparameters(ef)))

function isproper(::NaturalParametersSpace, ::Type{Multinomial}, η, conditioner)
    return conditioner isa Integer && conditioner >= 0 && !isempty(η) &&
           all(x -> !isnan(x) && x < Inf, η) && any(isfinite, η)
end

function isproper(::DefaultParametersSpace, ::Type{Multinomial}, p, conditioner)
    return conditioner isa Integer && conditioner >= 0 && !isempty(p) &&
           all(x -> isfinite(x) && x >= 0, p) && isapprox(sum(p), one(eltype(p)))
end

unpack_parameters(::Type{Multinomial}, packed, conditioner) = (packed,)
unpack_parameters(::Type{Multinomial}, packed) = (packed,)

function separate_conditioner(::Type{Multinomial}, parameters)
    ntrials, p = parameters
    return ((p,), ntrials)
end

join_conditioner(::Type{Multinomial}, cparams, conditioner) = (conditioner, only(cparams))

function _multinomial_reference(p)
    reference = findlast(>(zero(eltype(p))), p)
    isnothing(reference) && throw(ArgumentError("Multinomial probabilities must have positive total mass."))
    return reference
end

function (::MeanToNatural{Multinomial})(parameters::Tuple{<:AbstractVector}, conditioner)
    (p,) = parameters
    logreference = log(p[_multinomial_reference(p)])
    return (map(pi -> log(pi) - logreference, p),)
end

function (::NaturalToMean{Multinomial})(parameters::Tuple{<:AbstractVector}, conditioner)
    return (softmax(only(parameters)),)
end

getsufficientstatistics(::Type{Multinomial}, _) = (identity,)
isbasemeasureconstant(::Type{Multinomial}) = NonConstantBaseMeasure()

# Keep R a static parameter: capturing a local DataType in the reduction's
# closure loses inference on Julia 1.10.
_multinomial_logbasemeasure(n, x) = _multinomial_logbasemeasure(promote_type(Float64, eltype(x)), n, x)
function _multinomial_logbasemeasure(::Type{R}, n, x) where {R}
    return loggamma(R(n) + 1) - sum(xi -> loggamma(R(xi) + 1), x)
end

_multinomial_basemeasure(n, x) = _multinomial_basemeasure(promote_type(Float64, eltype(x)), n, x)
function _multinomial_basemeasure(::Type{R}, n, x) where {R}
    # Gamma ratios retain exact small coefficients, including 5!/(1!2!2!) = 30.
    # Beyond this range even gamma(n + 1) overflows, although the ratio may not.
    if n <= 170
        return gamma(R(n) + 1) / prod(xi -> gamma(R(xi) + 1), x)
    end
    return exp(_multinomial_logbasemeasure(R, n, x))
end

getbasemeasure(::Type{Multinomial}, n) = Base.Fix1(_multinomial_basemeasure, n)
getlogbasemeasure(::Type{Multinomial}, n) = Base.Fix1(_multinomial_logbasemeasure, n)

# Interpret a zero count times a log probability of -Inf as zero.
function _multinomial_dot(η, x)
    result = zero(promote_type(eltype(η), eltype(x)))
    for i in eachindex(η, x)
        if !iszero(x[i])
            result += η[i] * x[i]
        end
    end
    return result
end

_scalarproduct(::Type{Multinomial}, η, statistics, conditioner) = _multinomial_dot(η, only(statistics))

getlogpartition(::NaturalParametersSpace, ::Type{Multinomial}, n::Int) = (η) -> n * logsumexp(η)
getgradlogpartition(::NaturalParametersSpace, ::Type{Multinomial}, n::Int) = (η) -> n .* softmax(η)
getfisherinformation(::NaturalParametersSpace, ::Type{Multinomial}, n::Int) = (η) -> begin
    p = softmax(η)
    return n * (Diagonal(p) - p * p')
end

# Keeping log(sum(p)) makes derivatives agree with the log-ratio mapping even
# off the simplex. The reference is the last category in the interior.
getlogpartition(::DefaultParametersSpace, ::Type{Multinomial}, n::Int) = (p) ->
    n * (log(sum(p)) - log(p[_multinomial_reference(p)]))

getgradlogpartition(::DefaultParametersSpace, ::Type{Multinomial}, n::Int) =
    (p) -> begin
        reference = _multinomial_reference(p)
        common = n / sum(p)
        return map(i -> common - (i == reference ? n / p[reference] : zero(common)), eachindex(p))
    end

getfisherinformation(::DefaultParametersSpace, ::Type{Multinomial}, n::Int) =
    (p) -> begin
        all(>(zero(eltype(p))), p) || throw(ArgumentError("Default-space Multinomial Fisher information requires positive probabilities."))
        total = sum(p)
        return (n / total) * Diagonal(map(inv, p)) - fill(n / total^2, length(p), length(p))
    end

_fishermul_storage(::Type{Multinomial}, c, η, ::Type{R}) where {R} = nothing
function _fishermul_kernel!(w, ::Type{Multinomial}, n, η, v, ::Nothing)
    softmax!(w, η)
    weighted = dot(w, v)
    @inbounds for i in eachindex(w)
        w[i] *= n * (v[i] - weighted)
    end
    return w
end

struct _MultinomialProductLogPartition
    categories::Int
    trials::Int
    logfactorials::Vector{Float64}
end

_multinomial_primal_type(::Type{T}) where {T} = T
_multinomial_primal_type(::Type{<:ForwardDiff.Dual{Tag, V}}) where {Tag, V} = _multinomial_primal_type(V)

"""
    __compute_logpartition_multinomial_product(K, n)

Return the exact log normalizer for the product of two Multinomials with `K`
categories and `n` trials. A log-space convolution costs `O(K*n^2)` time and
`O(n)` workspace per evaluation. The cached log-factorials are read-only.

This is the Conway–Maxwell–multinomial normalizer with dispersion 2: Morris,
Raim & Sellers (2020), §3, equations (3)–(4), doi:10.1016/j.jmva.2020.104651.
See [Multinomial products](@ref examples-multinomial-product) for the recurrence's
derivation, numerical behavior, and literature references.
"""
function __compute_logpartition_multinomial_product(K::Int, n::Int)
    K >= 1 || throw(ArgumentError("A Multinomial must have at least one category."))
    n >= 0 || throw(ArgumentError("The number of trials must be nonnegative."))
    return _MultinomialProductLogPartition(K, n, [loggamma(m + 1) for m in 0:n])
end

function (partition::_MultinomialProductLogPartition)(η)
    length(η) == partition.categories || throw(DimensionMismatch("The Multinomial product has an incompatible parameter length."))
    all(x -> !isnan(x) && x < Inf, η) || throw(ArgumentError("Multinomial log weights must be finite or -Inf."))
    R = float(eltype(η))
    n = partition.trials
    iszero(n) && return zero(R)
    firstactive = findfirst(isfinite, η)
    isnothing(firstactive) && return R(-Inf)

    # These constants do not depend on η. Keep them outside AD, at the underlying
    # precision, including when computing nested derivatives with BigFloat.
    V = _multinomial_primal_type(R)
    logfactorials = V === Float64 ? partition.logfactorials : [loggamma(V(m + 1)) for m in 0:n]
    previous = Vector{R}(undef, n + 1)
    current = similar(previous)
    previous[1] = zero(R)
    for m in 1:n
        previous[m+1] = m * η[firstactive]
    end
    for j in eachindex(η)
        # A category with zero probability can only receive zero counts.
        (j == firstactive || η[j] == -Inf) && continue
        for m in 0:n
            value = previous[m+1]
            for r in 1:m
                logchoose = logfactorials[m+1] - logfactorials[r+1] - logfactorials[m-r+1]
                term = 2 * logchoose + r * η[j] + previous[m-r+1]
                value = logaddexp(value, term)
            end
            current[m+1] = value
        end
        previous, current = current, previous
    end
    return previous[n+1]
end

const _MultinomialProductAttributes =
    ExponentialFamilyDistributionAttributes{B, S, _MultinomialProductLogPartition, _MultinomialSupport, LB, D} where {B, S, LB, D}

# Specialize only the product's attributes, leaving other generic families alone.
getgradlogpartition(attributes::_MultinomialProductAttributes, ::ExponentialFamilyDistribution) =
    (η) -> ForwardDiff.gradient(getlogpartition(attributes), η)
getfisherinformation(attributes::_MultinomialProductAttributes, ::ExponentialFamilyDistribution) =
    (η) -> ForwardDiff.hessian(getlogpartition(attributes), η)

function _plogpdf(ef::ExponentialFamilyDistribution{Multivariate, P, C, A}, x, logpartition, logbasemeasure) where {P, C, A <: _MultinomialProductAttributes}
    @assert insupport(ef, x) lazy"Point $(x) does not belong to the support of $(ef)"
    return logbasemeasure + _multinomial_dot(getnaturalparameters(ef), x) - logpartition
end

# The product has a squared base measure and is not itself a Multinomial.
function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{Multinomial},
    right::ExponentialFamilyDistribution{Multinomial}
)
    n = getconditioner(left)
    @assert n == getconditioner(right) "$(left) and $(right) must have the same conditioner"
    ηleft, ηright = getnaturalparameters(left), getnaturalparameters(right)
    length(ηleft) == length(ηright) || throw(DimensionMismatch("Multinomial products require the same number of categories."))
    η = ηleft + ηright
    (iszero(n) || any(isfinite, η)) || throw(ArgumentError("The Multinomial product has zero total mass."))
    K = length(η)
    logbasemeasure = (x) -> 2 * _multinomial_logbasemeasure(n, x)
    basemeasure = (x) -> exp(logbasemeasure(x))
    attributes = ExponentialFamilyDistributionAttributes(
        basemeasure, (identity,), __compute_logpartition_multinomial_product(K, n),
        _MultinomialSupport(n, K), logbasemeasure, nothing
    )
    return ExponentialFamilyDistribution(Multivariate, η, n, attributes)
end
