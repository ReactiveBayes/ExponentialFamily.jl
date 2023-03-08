export Binomial
export BinomiallNaturalParameters, lognormalizer, naturalparams, logpdf_sample_friendly

import Distributions: Binomial, ntrials, succprob, failprob

vague(::Type{<:Binomial}, trials::Int) = Binomial(n)

prod_analytical_rule(::Type{<:Binomial}, ::Type{<:Binomial}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Binomial, right::Binomial)
    
    @assert ntrials(left) == ntrials(right) "Number of trials in $(left) and $(right) is not equal"
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Binomial(pprod / norm)
end

probvec(dist::Binomial) = (failprob(dist), succprob(dist))

struct BinomialNaturalParameters{T <: Real} <: NaturalParameters
    η::T
end

function Base.vec(p::BinomialNaturalParameters)
    return [p.η]
end

function BinomialNaturalParameters(v::AbstractVector)
    @assert length(v) === 1 "`BinomialNaturalParameters` must accept a vector of length `1`."
    return BinomialNaturalParameters(v[1])
end

Base.convert(::Type{BinomialNaturalParameters}, η::Real) = convert(BinomialNaturalParameters{typeof(η)}, η)

Base.convert(::Type{BinomialNaturalParameters{T}}, η::Real) where {T} = BinomialNaturalParameters(convert(T, η))

Base.convert(::Type{BinomialNaturalParameters}, vec::AbstractVector) =
    convert(BinomialNaturalParameters{eltype(vec)}, vec)

Base.convert(::Type{BinomialNaturalParameters{T}}, vec::AbstractVector) where {T} =
    BinomialNaturalParameters(convert(AbstractVector{T}, vec))

function Base.:(==)(left::BinomialNaturalParameters, right::BinomialNaturalParameters)
    return left.η == right.η
end

as_naturalparams(::Type{T}, args...) where {T <: BinomialNaturalParameters} =
    convert(BinomialNaturalParameters, args...)

# Standard parameters to natural parameters
function naturalparams(dist::Binomial)
    @assert !(succprob(dist) ≈ 1) "Binomial natural parameters are not defiend for p = 1."
    return BinomialNaturalParameters(log(succprob(dist) / (1 - succprob(dist))))
end

function convert(::Type{Distribution}, p::BinomialNaturalParameters)
    return Binomial(n, exp(params.η) / (1 + exp(params.η)))
end

function Base.:+(left::BinomialNaturalParameters, right::BinomialNaturalParameters)
    return BinomialNaturalParameters(left.η .+ right.η)
end

function Base.:-(left::BinomialNaturalParameters, right::BinomialNaturalParameters)
    return BinomialNaturalParameters(left.η .- right.η)
end

function lognormalizer(params::BinomialNaturalParameters)
    return -log(logistic(-params.η))
end

function Distributions.logpdf(params::BinomialNaturalParameters, x)
    return x * params.η - lognormalizer(params)
end