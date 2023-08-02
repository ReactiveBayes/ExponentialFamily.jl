export Categorical
export logpartition

import Distributions: Categorical, probs
import LogExpFunctions: logsumexp
import FillArrays: OneElement
using LoopVectorization

vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)

default_prod_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = ClosedProd()

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
    return LoopVectorization.vmap(d -> log(d / p[end]), p)
end

unpack_naturalparameters(ef::ExponentialFamilyDistribution{Categorical}) = (getnaturalparameters(ef),)

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Categorical) =
    ExponentialFamilyDistribution(Categorical, pack_naturalparameters(dist))

Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Categorical}) =
    Categorical(softmax(getnaturalparameters(exponentialfamily)))

check_valid_natural(::Type{<:Categorical}, params) = first(size(params)) >= 2

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Categorical})
    logsumexp(getnaturalparameters(exponentialfamily))
end
isproper(::ExponentialFamilyDistribution{Categorical}) = true

function fisherinformation(expfamily::ExponentialFamilyDistribution{Categorical})
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

function support(ef::ExponentialFamilyDistribution{Categorical})
    return ClosedInterval{Int}(1, length(getnaturalparameters(ef)))
end

function insupport(ef::ExponentialFamilyDistribution{Categorical, P, C, Safe}, x::Real) where {P, C}
    return x ∈ support(ef)
end

function insupport(union::ExponentialFamilyDistribution{Categorical, P, C, Safe}, x::Vector) where {P, C}
    return typeof(x) <: Vector{<:Integer} && sum(x) == 1 && length(x) == maximum(support(union))
end

basemeasureconstant(::ExponentialFamilyDistribution{Categorical}) = ConstantBaseMeasure()
basemeasureconstant(::Type{<:Categorical}) = ConstantBaseMeasure()

basemeasure(::Type{<:Categorical}) = one(Float64)
basemeasure(::ExponentialFamilyDistribution{Categorical}) = one(Float64)
basemeasure(::ExponentialFamilyDistribution{Categorical}, x::Real) = one(x)
basemeasure(::ExponentialFamilyDistribution{Categorical}, x::Vector) = one(eltype(x))

sufficientstatistics(ef::ExponentialFamilyDistribution{<:Categorical}) = x -> sufficientstatistics(ef, x)
sufficientstatistics(ef::ExponentialFamilyDistribution{<:Categorical}, x::Real) =
    OneElement(x, length(getnaturalparameters(ef)))
sufficientstatistics(::ExponentialFamilyDistribution{<:Categorical}, x::Vector) = x
