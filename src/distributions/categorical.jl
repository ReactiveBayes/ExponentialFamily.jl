export Categorical
export CategoricalNaturalParameters, lognormalizer, naturalparams, logpdf_sample_friendly

import Distributions: Categorical, probs

vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)

prod_analytical_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = ProdAnalyticalRuleAvailable()

convert_eltype(::Type{Categorical}, ::Type{T}, distribution::Categorical{R}) where {T <: Real, R <: Real} =
    Categorical(convert(AbstractVector{T}, probs(distribution)))

function Base.prod(::ProdAnalytical, left::Categorical, right::Categorical)
    # Multiplication of 2 categorical PMFs: p(z) = p(x) * p(y)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)
    return Categorical(mvec ./ norm)
end

probvec(dist::Categorical) = probs(dist)

function compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Categorical)
    return log(dot(probvec(left_dist), probvec(right_dist)))
end

struct CategoricalNaturalParameters{T <: Real} <: NaturalParameters
    logprobabilities::Array{T}
end

function CategoricalNaturalParameters(logprobabilities::AbstractVector{<:Real})
    T = promote_type(eltype(logprobabilities))
    return CategoricalNaturalParameters(convert(AbstractArray{T}, logprobabilities))
end

function CategoricalNaturalParameters(logproabilities::AbstractVector{<:Integer})
    return CategoricalNaturalParameters(float.(logproabilities))
end

function CategoricalNaturalParameters(logprobabilities::AbstractVector{T}) where {T}
    return CategoricalNaturalParameters{T}(logprobabilities)
end

Base.convert(::Type{CategoricalNaturalParameters}, logprobabilities::AbstractVector) =
    convert(CategoricalNaturalParameters{promote_type(eltype(logprobabilities))}, logprobabilities)

Base.convert(::Type{CategoricalNaturalParameters{T}}, logprobabilities::AbstractVector) where {T} =
    CategoricalNaturalParameters(convert(AbstractVector{T}, logprobabilities))

Base.convert(::Type{CategoricalNaturalParameters}, vector::AbstractVector) =
    convert(CategoricalNaturalParameters{eltype(vector)}, vector)

Base.convert(::Type{CategoricalNaturalParameters{T}}, vector::AbstractVector) where {T} =
    CategoricalNaturalParameters(convert(AbstractVector{T}, vector))

function Base.:(==)(left::CategoricalNaturalParameters, right::CategoricalNaturalParameters)
    return left.logprobabilities == right.logprobabilities
end

as_naturalparams(::Type{T}, args...) where {T <: CategoricalNaturalParameters} =
    convert(CategoricalNaturalParameters, args...)

function Base.vec(p::CategoricalNaturalParameters)
    return p
end

# Standard parameters to natural parameters
function naturalparams(dist::Categorical)
    logprobabilities = log.(probvec(dist))
    return CategoricalNaturalParameters(logprobabilities)
end

function convert(::Type{Distribution}, η::CategoricalNaturalParameters)
    return Categorical(softmax(η.logprobabilities))
end

function Base.:+(left::CategoricalNaturalParameters, right::CategoricalNaturalParameters)
    return CategoricalNaturalParameters(left.logprobabilities .+ right.logprobabilities)
end

function Base.:-(left::CategoricalNaturalParameters, right::CategoricalNaturalParameters)
    return CategoricalNaturalParameters(left.logprobabilities .- right.logprobabilities)
end

function lognormalizer(::CategoricalNaturalParameters)
    return 0.0
end

function Distributions.logpdf(η::CategoricalNaturalParameters, x)
    return Distributions.logpdf(convert(Categorical, η), x)
end

isproper(params::CategoricalNaturalParameters) = true

function logpdf_sample_friendly(dist::Categorical)
    p = probvec(dist)
    friendly = Categorical(p)
    return (friendly, friendly)
end
