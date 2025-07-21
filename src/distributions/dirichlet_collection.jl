export DirichletCollection

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf
using Distributions
using SpecialFunctions, LogExpFunctions

import FillArrays: Ones, Eye
using LinearAlgebra, Random

using BlockArrays: BlockDiagonal

"""
    DirichletCollection{T <: Real, N, A <: AbstractArray{T, N}} <: Distribution{ArrayLikeVariate{N}, Continuous}

A collection of independent Dirichlet distributions, where `T` is the element type of the tensor `A`. The distribution generalizes the Dirichlet distribution to handle multiple sets of parameters organized in a tensor structure. This distribution collects multiple independent Dirichlet distributions into a single efficient interface. The Dirichlet counts for the independent Dirichlet distributions are stored along the first dimension of `a`. This distribution can be used as a conjugate prior to a Categorical distribution with mulitple switch cases (such as a discrete state-transition with controls).  

# Fields
- `α::A`: The tensor parameter of the distribution, where each slice represents parameters of a Dirichlet distribution
- `α0::A`: The sum of parameters along the first dimension.
- `lmnB::A`: The log multinomial beta function values for each slice.

The distribution models multiple independent Dirichlet distributions organized in a tensor structure, where each slice `a[:,i,j,...]` represents the parameters of an independent Dirichlet distribution.
"""
struct DirichletCollection{T <: Real, N, A <: AbstractArray{T, N}} <: Distribution{ArrayLikeVariate{N}, Continuous}
    α::A
    α0::A
    lmnB::A
    function DirichletCollection(alpha::AbstractArray{T, N}) where {T, N}
        if !all(x -> x > zero(x), alpha)
            throw(ArgumentError("All elements of the alpha tensor should be positive"))
        end
        alpha0 = sum(alpha; dims = 1)
        lmnB = sum(loggamma, alpha; dims = 1) - loggamma.(alpha0)
        NT = promote_type(T, eltype(alpha0), eltype(lmnB))
        alpha = convert_paramfloattype(NT, alpha)
        alpha0 = convert_paramfloattype(NT, alpha0)
        lmnB = convert_paramfloattype(NT, lmnB)
        new{NT, N, typeof(alpha)}(alpha, alpha0, lmnB)
    end
end

function BayesBase.logpdf(dist::DirichletCollection{T, N, A}, xs::AbstractVector{A}) where {T, N, A}
    return map(x -> logpdf(dist, x), xs)
end

function BayesBase.pdf(dist::DirichletCollection{R, N, A}, x::AbstractArray{T, N}) where {R, A, T <: Real, N}
    return exp(logpdf(dist, x))
end

function BayesBase.pdf(dist::DirichletCollection, xs::AbstractVector)
    return map(x -> pdf(dist, x), xs)
end

BayesBase.params(dist::DirichletCollection) = (dist.α,)

function unpack_parameters(::Type{DirichletCollection}, packed, conditioner)
    packed = view(packed, 1:length(packed))
    return (reshape(packed, conditioner),)
end

function join_conditioner(::Type{DirichletCollection}, cparams, _)
    return cparams
end

function separate_conditioner(::Type{DirichletCollection}, tuple_of_θ)
    return (tuple_of_θ, size(tuple_of_θ[1]))
end

isbasemeasureconstant(::Type{DirichletCollection}) = ConstantBaseMeasure()

getbasemeasure(::Type{DirichletCollection}, conditioner) = (x) -> one(Float64)
getlogbasemeasure(::Type{DirichletCollection}, conditioner) = (x) -> zero(Float64)

getsufficientstatistics(::Type{DirichletCollection}, conditioner) = (x -> vmap(log, x),)

BayesBase.mean(dist::DirichletCollection) = dist.α ./ dist.α0
BayesBase.mean(::BroadcastFunction{typeof(log)}, dist::DirichletCollection) = digamma.(dist.α) .- digamma.(dist.α0)
BayesBase.mean(::BroadcastFunction{typeof(clamplog)}, dist::DirichletCollection{T, N, A}) where {T, N, A} =
    digamma.(clamp.(dist.α, tiny, typemax(T))) .- digamma.(dist.α0)
function BayesBase.cov(dist::DirichletCollection{T}) where {T}
    s = size(dist.α)
    news = (first(s), first(s), Base.tail(s)...)
    v = zeros(T, news)
    for i in CartesianIndices(Base.tail(size(dist.α)))
        v[:, :, i] .= cov(Dirichlet(dist.α[:, i]))
    end
    return v
end

function BayesBase.var(dist::DirichletCollection{T, N, A}) where {T, N, A}
    α = dist.α
    α0 = dist.α0
    c = inv.(α0 .^ 2 .* (α0 .+ 1))
    v = α .* (α0 .- α) .* c
    return v
end

BayesBase.std(dist::DirichletCollection) = sqrt.(var(dist))

Base.size(dist::DirichletCollection) = size(dist.α)
Base.eltype(::DirichletCollection{T}) where {T} = T

function BayesBase.vague(::Type{<:DirichletCollection}, dims::Tuple)
    return DirichletCollection(ones(Float64, dims))
end

function BayesBase.entropy(dist::DirichletCollection)
    α = dist.α
    α0 = dist.α0
    lmnB = dist.lmnB
    return sum(-sum((α .- one(eltype(α))) .* (digamma.(α) .- digamma.(α0)); dims = 1) .+ lmnB)
end

function BayesBase.rand(rng::AbstractRNG, dist::DirichletCollection{T}) where {T}
    container = similar(dist.α)
    return rand!(rng, dist, container)
end

function BayesBase.rand(rng::AbstractRNG, dist::DirichletCollection{T}, nsamples::Int64) where {T}
    container = [similar(dist.α) for _ in 1:nsamples]
    rand!(rng, dist, container)
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::DirichletCollection, container::AbstractArray{T, N}) where {T <: Real, N}
    for (i, αi) in zip(eachindex(container), dist.α)
        @inbounds container[i] = rand(rng, Gamma(αi))
    end
    container .= container ./ sum(container; dims = 1)
end

# Add method for handling vector of arrays
function BayesBase.rand!(
    rng::AbstractRNG,
    dist::DirichletCollection{T, N, A},
    container::AbstractArray{A, M}
) where {T <: Real, N, A <: AbstractArray{T, N}, M}
    for c in container
        size(c) == size(dist.α) || error("Size mismatch")
    end

    @inbounds for c in container
        rand!(rng, dist, c)
    end

    return container
end

function BayesBase.logpdf(dist::DirichletCollection{R, N, A}, x::AbstractArray{T, N}) where {R, A, T <: Real, N}
    if !insupport(dist, x)
        return sum(xlogy.(one(eltype(dist.α)), zero(eltype(x))))
    end
    α = dist.α
    α0 = dist.α0
    s = sum(xlogy.(α .- 1, x); dims = 1)
    return sum(s .- dist.lmnB)
end

check_logpdf(::ExponentialFamilyDistribution{DirichletCollection}, x::AbstractVector) = (MapBasedLogpdfCall(), x)
check_logpdf(::ExponentialFamilyDistribution{DirichletCollection}, x) = (PointBasedLogpdfCall(), x)

BayesBase.default_prod_rule(::Type{<:DirichletCollection}, ::Type{<:DirichletCollection}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::DirichletCollection, right::DirichletCollection)
    return DirichletCollection(left.α .+ right.α .- 1)
end

function BayesBase.insupport(dist::DirichletCollection{T, N, A}, x::AbstractArray{T, N}) where {T, N, A}
    return size(dist) == size(x) && !any(x -> x < zero(x), x) && all(z -> z ≈ 1, sum(x; dims = 1))
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{DirichletCollection}, x)
    l = getconditioner(ef)
    values = map(CartesianIndices(Base.tail(size(x)))) do i
        slice = @view x[:, i]
        sum(slice) ≈ 1 && all(y -> y > 0, slice)
    end
    return l == size(x) && all(values)
end

# Natural parametrization

function isproper(::NaturalParametersSpace, ::Type{DirichletCollection}, η, conditioner)
    return length(η) > 1 && all(isless.(-1, η)) && all(!isinf, η) && all(!isnan, η)
end
function isproper(::MeanParametersSpace, ::Type{DirichletCollection}, θ, conditioner)
    return length(θ) > 1 && all(>(0), θ) && all(!isinf, θ)
end

function (::MeanToNatural{<:DirichletCollection})(tuple_of_θ::Tuple{Any}, _)
    (α,) = tuple_of_θ
    return (α - Ones{Float64}(size(α)),)
end

function (::NaturalToMean{DirichletCollection})(tuple_of_η::Tuple{Any}, _)
    (η,) = tuple_of_η
    return (η + Ones{Float64}(size(η)),)
end

function getlogpartition(::NaturalParametersSpace, ::Type{DirichletCollection}, conditioner::NTuple{N, Int}) where {N}
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions
    dirichlet_logpartition = getlogpartition(NaturalParametersSpace(), Dirichlet)

    return function (η::AbstractVector)
        result = zero(eltype(η))
        for i in 1:n_distributions
            idx_start = (i - 1) * k + 1
            idx_end = i * k
            @views params = η[idx_start:idx_end]
            result += dirichlet_logpartition(params)
        end

        return result
    end
end

function getgradlogpartition(
    ::NaturalParametersSpace,
    ::Type{DirichletCollection},
    conditioner::NTuple{N, Int}
) where {N}
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions

    # Get the "gradlogpartition" function for a standard Dirichlet
    dirichlet_gradlogpartition = getgradlogpartition(NaturalParametersSpace(), Dirichlet)

    return function (η::AbstractVector{T}) where {T}
        # Preallocate the output. We know we need `k * n_distributions` entries,
        # of the same element type as `η`.
        out = Vector{T}(undef, k * n_distributions)

        for i in 1:n_distributions
            @inbounds begin
                # For the i-th distribution, grab the slice of η
                # and apply the Dirichlet gradlogpartition.
                out[(i-1)*k+1:i*k] = dirichlet_gradlogpartition(
                    @view η[(i-1)*k+1:i*k]
                )
            end
        end
        return out
    end
end

function getfisherinformation(::NaturalParametersSpace, ::Type{DirichletCollection}, conditioner)
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions
    dirichlet_fisher = getfisherinformation(NaturalParametersSpace(), Dirichlet)

    return function (η::AbstractVector)
        blocks = Vector{Matrix{Float64}}(undef, n_distributions)

        for i in 1:n_distributions
            idx_start = (i - 1) * k + 1
            idx_end = i * k
            @views params = η[idx_start:idx_end]
            blocks[i] = dirichlet_fisher(params)
        end

        return BlockDiagonal(blocks)
    end
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{DirichletCollection}, conditioner) =
    (η) -> begin
        return mapreduce(x -> getlogpartition(MeanParametersSpace(), Dirichlet)(x), +, η)
    end

function getgradlogpartition(::MeanParametersSpace, ::Type{DirichletCollection}, conditioner::NTuple{N, Int}) where {N}
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions
    dirichlet_gradlogpartition = getgradlogpartition(MeanParametersSpace(), Dirichlet)

    return function (θ::AbstractVector{T}) where {T}
        # Preallocate the output
        out = Vector{T}(undef, k * n_distributions)

        for i in 1:n_distributions
            @inbounds begin
                # For each distribution, compute its gradient
                out[(i-1)*k+1:i*k] = dirichlet_gradlogpartition(
                    @view θ[(i-1)*k+1:i*k]
                )
            end
        end
        return out
    end
end

function getfisherinformation(::MeanParametersSpace, ::Type{DirichletCollection}, conditioner::NTuple{N, Int}) where {N}
    k = conditioner[1]  # Number of parameters per distribution
    n_distributions = prod(Base.tail(conditioner))  # Total number of distributions
    dirichlet_fisher = getfisherinformation(MeanParametersSpace(), Dirichlet)

    return function (θ::AbstractVector{T}) where {T}
        # Create blocks for block diagonal matrix
        blocks = Vector{Matrix{Float64}}(undef, n_distributions)

        for i in 1:n_distributions
            @inbounds begin
                # For each distribution, compute its Fisher information
                blocks[i] = dirichlet_fisher(
                    @view θ[(i-1)*k+1:i*k]
                )
            end
        end

        return BlockDiagonal(blocks)
    end
end

_scalarproduct(::Type{DirichletCollection}, η, statistics, conditioner::NTuple{N, Int}) where {N} =
    _scalarproduct(ArrayLikeVariate{N}, DirichletCollection, η, statistics, conditioner)
