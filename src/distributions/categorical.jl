export Categorical
export logpartition

import Distributions: Categorical, probs
import LogExpFunctions: logsumexp
import FillArrays: OneElement

vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)

closed_prod_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = ClosedProd()

convert_eltype(::Type{Categorical}, ::Type{T}, distribution::Categorical{R}) where {T <: Real, R <: Real} =
    Categorical(convert(AbstractVector{T}, probs(distribution)))

function Base.prod(::ClosedProd, left::Categorical, right::Categorical)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)
    return Categorical(mvec ./ norm)
end

probvec(dist::Categorical) = probs(dist)

function compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Categorical)
    return log(dot(probvec(left_dist), probvec(right_dist)))
end

function pack_naturalparameters(dist::Categorical)
    p = probvec(dist)
    return log.(p/p[end])
end

Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Categorical) = KnownExponentialFamilyDistribution(Categorical, pack_naturalparameters(dist))

Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Categorical}) = Categorical(softmax(getnaturalparameters(exponentialfamily)))

check_valid_natural(::Type{<:Categorical}, params) = first(size(params)) >= 2

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Categorical}) 
    logsumexp(getnaturalparameters(exponentialfamily))
end
isproper(::KnownExponentialFamilyDistribution{Categorical}) = true

function fisherinformation(expfamily::KnownExponentialFamilyDistribution{Categorical})
    η = getnaturalparameters(expfamily)
    I = Matrix{Float64}(undef, length(η), length(η))
    @inbounds for i in 1:length(η)
        I[i, i] = exp(η[i]) * (sum(exp.(η)) - exp(η[i])) / (sum(exp.(η)))^2
        for j in 1:i-1
            I[i, j] = -exp(η[i]) * exp(η[j]) / (sum(exp.(η)))^2
            I[j, i] = I[i, j]
        end
    end
    return I
end

function fisherinformation(dist::Categorical)
    p = probvec(dist)
    I = Matrix{Float64}(undef, length(p), length(p))
    @inbounds for i in 1:length(p)
        I[i, i] = 1 / p[i]
        for j in 1:i-1
            I[i, j] = 0
            I[j, i] = I[i, j]
        end
    end
    return I
end

function support(ef::KnownExponentialFamilyDistribution{Categorical})
    return ClosedInterval{Int}(1, length(getnaturalparameters(ef)))
end

function insupport(ef::KnownExponentialFamilyDistribution{Categorical, P, C, Safe}, x::Real) where {P, C}
    return x ∈ support(ef)
end

function insupport(union::KnownExponentialFamilyDistribution{Categorical, P, C, Safe}, x::Vector) where {P, C}
    return typeof(x) <: Vector{<:Integer} && sum(x) == 1 && length(x) == maximum(support(union))
end

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Categorical}, <:Categorical}, x::Real)
    @assert insupport(union, x) "Evaluation point $(x) is not in the support of Categorical"
    return one(x)
end

function basemeasure(union::Union{<:KnownExponentialFamilyDistribution{Categorical}, <:Categorical}, x::Vector)
    @assert insupport(union, x) "Evaluation point $(x) is not in the support of Categorical"
    return one(eltype(x))
end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Categorical}, x::Real)
    @assert insupport(ef, x) "Evaluation point $(x) is not in the support of Categorical"
    return OneElement(x,length(getnaturalparameters(ef)))
end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Categorical}, x::Vector)
    @assert insupport(ef, x) "Evaluation point $(x) is not in the support of Categorical"
    return x
end
