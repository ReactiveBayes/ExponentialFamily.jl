export TensorDirichlet, ContinuousTensorDistribution

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf
using Distributions
using SpecialFunctions, LogExpFunctions

import FillArrays: Ones, Eye
import LoopVectorization: vmap, vmapreduce
using LinearAlgebra, Random

const ContinuousTensorDistribution = Distribution{ArrayLikeVariate, Continuous}

"""
    TensorDirichlet{T <: Real, A <: AbstractArrray{T,3}} <: ContinuousTensorDistribution

A tensor-valued TensorDirichlet distribution, where `T` is the element type of the tensor `A`.
The `a` field stores the matrix parameter of the distribution.

# Fields
- `a::A`: The matrix parameter of the TensorDirichlet distribution.

# Model
- a[:,m,n] are the parameters of a Dirichlet distribution
- a[:,m_1,n_1] and a[:,m_2,n_2] are supposed independent if (m_1,n_1) not equal to (m_2,n_2).
"""
struct TensorDirichlet{T <: Real, N, A <: AbstractArray{T, N}, Ts} <: ContinuousTensorDistribution
    a::A
    α0::Ts
    lmnB::Ts
    function TensorDirichlet(alpha::AbstractArray{T, N}) where {T, N}
        if !all(x -> x > zero(x), alpha)
            throw(ArgumentError("All elements of the alpha tensor should be positive"))
        end
        alpha0 = sum(alpha; dims = 1)
        lmnB = sum(loggamma, alpha; dims = 1) - loggamma.(alpha0)
        new{T, N, typeof(alpha), typeof(alpha0)}(alpha, alpha0, lmnB)
    end
end

get_dirichlet_parameters(dist::TensorDirichlet{T, N, A}) where {T, N, A} = eachslice(dist.a, dims = Tuple(2:N))
extract_collection(dist::TensorDirichlet) = [dist.a[:, i] for i in CartesianIndices(Base.tail(size(dist.a)))]
unpack_parameters(::Type{TensorDirichlet}, packed) = ([packed[:, i] for i in CartesianIndices(Base.tail(size(packed)))],)
BayesBase.params(::MeanParametersSpace, dist::TensorDirichlet) = (reduce(vcat, extract_collection(dist)),)
getbasemeasure(::Type{TensorDirichlet}) = (x) -> sum([x[:, i] for i in CartesianIndices(Base.tail(size(x)))])
getsufficientstatistics(::TensorDirichlet) = (x -> vmap(log, x),)

BayesBase.mean(dist::TensorDirichlet) = dist.a ./ dist.α0
function BayesBase.cov(dist::TensorDirichlet{T}) where {T}
    s = size(dist.a)
    news = (first(s), first(s), Base.tail(s)...)
    v = zeros(T, news)
    for i in CartesianIndices(Base.tail(size(dist.a)))
        v[:, :, i] .= cov(Dirichlet(dist.a[:, i]))
    end
    return v
end
function BayesBase.var(dist::TensorDirichlet{T, N, A, Ts}) where {T, N, A, Ts}
    α = dist.a
    α0 = dist.α0
    c = inv.(α0 .^ 2 .* (α0 .+ 1))
    v = α .* (α0 .- α) .* c
    return v
end
BayesBase.std(dist::TensorDirichlet) = map(d -> std(Dirichlet(d)), extract_collection(dist))

BayesBase.params(dist::TensorDirichlet) = (dist.a,)

Base.size(dist::TensorDirichlet) = size(dist.a)
Base.eltype(::TensorDirichlet{T}) where {T} = T

function BayesBase.vague(::Type{<:TensorDirichlet}, dims::Int)
    return TensorDirichlet(ones(dims, dims))
end

function BayesBase.vague(::Type{<:TensorDirichlet}, dims::Tuple)
    return TensorDirichlet(ones(Float64, dims))
end

function BayesBase.entropy(dist::TensorDirichlet)
    return vmapreduce(+, extract_collection(dist)) do column
        scolumn = sum(column)
        -sum((column .- one(Float64)) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) +
        sum(loggamma.(column))
    end
end

BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:TensorDirichlet}) = TensorDirichlet
BayesBase.promote_variate_type(::Type{ArrayLikeVariate}, ::Type{<:Dirichlet}) = TensorDirichlet

function BayesBase.rand(rng::AbstractRNG, dist::TensorDirichlet{T}) where {T}
    container = similar(dist.a)
    return rand!(rng, dist, container)
end

function BayesBase.rand(rng::AbstractRNG, dist::TensorDirichlet{T}, nsamples::Int64) where {T}
    container = Vector{typeof(dist.a)}(undef, nsamples)
    @inbounds for i in eachindex(container)
        container[i] = similar(dist.a)
        rand!(rng, dist, container[i])
    end
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::TensorDirichlet, container::AbstractArray{T, N}) where {T <: Real, N}
    for index in CartesianIndices(extract_collection(dist))
        rand!(rng, Dirichlet(dist.a[:, index]), @view container[:, index])
    end
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::TensorDirichlet{A}, container::AbstractArray{A, N}) where {T <: Real, N, A <: AbstractArray{T, N}}
    for i in container
        rand!(rng, dist, @view container[i])
    end
    return container
end

function BayesBase.logpdf(dist::TensorDirichlet{R, N, A}, x::AbstractArray{T, N}) where {R, A, T <: Real, N}
    α = dist.a
    α0 = dist.α0
    s = sum(xlogy.(α .- 1, x); dims = 1)
    return sum(s .- dist.lmnB)
end

BayesBase.pdf(dist::TensorDirichlet, x::Array{T, N}) where {T <: Real, N} = exp(logpdf(dist, x))

BayesBase.default_prod_rule(::Type{<:TensorDirichlet}, ::Type{<:TensorDirichlet}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::TensorDirichlet, right::TensorDirichlet)
    return TensorDirichlet(left.a .+ right.a .- 1)
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{TensorDirichlet}, x)
    l = size(getnaturalparameters(ef))
    values = [x[:, i] for i in CartesianIndices(Base.tail(size(x)))]
    ## The element of the array should be the a categorical distribution (an vector of postive value that sum to 1)
    ## and all catagorical distribution should have the same size than the corresponding disrichlet prior (not checked).
    return l == size(x) && all(x -> sum(x) ≈ 1, values) && all(!any(x -> x < 0), values)
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{TensorDirichlet}, η, conditioner) =
    isnothing(conditioner) && length(η) > 1 && all(map(x -> isproper(NaturalParametersSpace(), Dirichlet, x), eachslice(η, dims = Tuple(2:ndims(η)))))
isproper(::MeanParametersSpace, ::Type{TensorDirichlet}, θ, conditioner) =
    isnothing(conditioner) && length(θ) > 1 && all(map(x -> isproper(MeanParametersSpace(), Dirichlet, x), eachslice(θ, dims = Tuple(2:ndims(θ)))))
isproper(p, ::Type{TensorDirichlet}, η, conditioner) =
    isnothing(conditioner) && all(x -> isproper(p, Type{Dirichlet}, x), unpack_parameters(TensorDirichlet, η))

function (::MeanToNatural{TensorDirichlet})(tuple_of_θ::Tuple{Any})
    (α,) = tuple_of_θ
    out = copy(α)
    for i in CartesianIndices(Base.tail(size(α)))
        out[:, i] = α[:, i] - ones(length(α[:, i]))
    end
    return out
end

function (::NaturalToMean{TensorDirichlet})(tuple_of_η::Tuple{Any})
    (α,) = tuple_of_η
    out = copy(α)
    for i in CartesianIndices(Base.tail(size(α)))
        out[:, i] = α[:, i] + ones(length(α[:, i]))
    end
    return out
end

getlogpartition(::NaturalParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return mapreduce(x -> getlogpartition(NaturalParametersSpace(), Dirichlet)(x), +, η)
    end

getgradlogpartition(::NaturalParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return map(d -> getgradlogpartition(NaturalParametersSpace(), Dirichlet)(d), η)
    end

getfisherinformation(::NaturalParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return mapreduce(d -> getfisherinformation(NaturalParametersSpace(), Dirichlet)(d), +, η)
    end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return mapreduce(x -> getlogpartition(MeanParametersSpace(), Dirichlet)(x), +, η)
    end

getgradlogpartition(::MeanParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return map(d -> getgradlogpartition(MeanParametersSpace(), Dirichlet)(d), η)
    end

getfisherinformation(::MeanParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return mapreduce(d -> getfisherinformation(MeanParametersSpace(), Dirichlet)(d), +, η)
    end
