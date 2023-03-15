export Contingency, icdf

using LinearAlgebra
using Random

"""
    Contingency(P, renormalize = Val(true))

The contingency distribution is a multivariate generalization of the categorical distribution. As a bivariate distribution, the 
contingency distribution defines the joint probability over two unit vectors `v1` and `v2` with one hot encoding. Or it can be thought of
as the joint distribution of two categoricals with supports `v1 ={ 1,2,...,N}` and `v2 ={ 1,2,...,N}`. 
The parameter `P` encodes a contingency matrix that specifies the probability of co-occurrence.

    v1 ∈ {0, 1}^d1 where Σ_j v1_j = 1
    v2 ∈ {0, 1}^d2 where Σ_k v2_k = 1

    P ∈ [0, 1]^{d1 × d2}, where Σ_jk P_jk = 1

    f(v1, v2, P) = Contingency(out1, out2 | P) = Π_jk P_jk^{v1_j * v2_k}

A `Contingency` distribution over more than two variables requires higher-order tensors as parameters; these are not implemented in ReactiveMP.

# Arguments:
- `P`, required, contingency matrix
- `renormalize`, optional, supports either `Val(true)` or `Val(false)`, specifies whether matrix `P` must be automatically renormalized. Does not modify the original `P` and allocates a new one for the renormalized version. If set to `false` the contingency matrix `P` **must** be normalized by hand, otherwise the result of related calculations might be wrong

"""
struct Contingency{T, P <: AbstractMatrix{T}} <: DiscreteMultivariateDistribution
    p::P

    Contingency{T, P}(A::AbstractMatrix) where {T, P <: AbstractMatrix{T}} = new(A)
end

Contingency(P::AbstractMatrix)                                               = Contingency(P, Val(true))
Contingency(P::M, renormalize::Val{true}) where {T, M <: AbstractMatrix{T}}  = Contingency{T, M}(P ./ sum(P))
Contingency(P::M, renormalize::Val{false}) where {T, M <: AbstractMatrix{T}} = Contingency{T, M}(P)

contingency_matrix(distribution::Contingency) = distribution.p

vague(::Type{<:Contingency}, dims::Int) = Contingency(ones(dims, dims) ./ abs2(dims))

convert_eltype(::Type{Contingency}, ::Type{T}, distribution::Contingency{R}) where {T <: Real, R <: Real} =
    Contingency(convert(AbstractArray{T}, contingency_matrix(distribution)))

## dispatch support is regular {1,2,..., N}
function pdf_contingency(distribution::Contingency, x::AbstractArray, T)
    @assert length(x) === 2 "$(x) should be length 2 vector with the entries corresponding to elements of the contingency matrix of $(distribution)"
    contingencymatrix = contingency_matrix(distribution)
    dim               = getindex(size(contingencymatrix), 1)
    support           = collect(1:getindex(size(contingencymatrix), 1))
    idx1              = searchsortedfirst(support, x[1])
    idx2              = searchsortedfirst(support, x[2])
    if idx1 <= dim && support[idx1] == x[1] && idx2 <= dim && support[idx2] == x[2]
        return contingencymatrix[idx1, idx2]
    else
        return zero(eltype(contingencymatrix))
    end
end

## dispatch when support is one hot coded
function pdf_contingency(distribution::Contingency, x::AbstractArray, T::Type{Bool})
    contingencymatrix = contingency_matrix(distribution)
    dim               = getindex(size(contingencymatrix), 1)
    @assert eltype(x) === Bool "Entries of one hot encoded vector should be boolean"
    @assert size(x) === (2, dim) "one hot encoded $(x) should be of same length with the entries corresponding to elements of the contingency matrix of $(distribution)"
    @assert all(map(sum, eachrow(x)) .=== 1) "Entries of one hot encoded vector should sum to 1"
    xconverted = [first(indexin(true, x[1, :])), first(indexin(true, x[2, :]))]
    return pdf(distribution, xconverted)
end

Distributions.pdf(distribution::Contingency, x::AbstractArray{T}) where {T <: Real} =
    pdf_contingency(distribution, x, eltype(x))

function Distributions.logpdf(distribution::Contingency, x::AbstractArray{T}) where {T <: Real}
    return log(Distributions.pdf(distribution, x))
end

function Distributions.mean(distribution::Contingency)
    contingency = contingency_matrix(distribution)
    support     = collect(1:length(contingency))

    return sum(collect(x) .* pdf(distribution, collect(x)) for x in Iterators.product(support, support))
end

function Distributions.cov(distribution::Contingency)
    contingencymatrix = contingency_matrix(distribution)
    dim = getindex(size(contingencymatrix), 1)
    support = collect(1:dim)
    return sum(
        (collect(x) - mean(distribution)) * (collect(x) - mean(distribution))' .* pdf(distribution, collect(x)) for
        x in Iterators.product(support, support)
    )
end

Distributions.var(distribution::Contingency) = diag(cov(distribution))

function entropy(distribution::Contingency)
    P = contingency_matrix(distribution)
    return -mapreduce((p) -> p * clamplog(p), +, P)
end

# Standard parameters to natural parameters
function Base.convert(::Type{NaturalParameters}, dist::Contingency)
    logcontingency = log.(contingency_matrix(dist))
    return NaturalParameters(Contingency, logcontingency)
end

function Base.convert(::Type{Distribution}, η::NaturalParameters{Contingency})
    return Contingency(softmax(get_params(η)))
end

function lognormalizer(::NaturalParameters{Contingency})
    return 0.0
end

check_valid_natural(::Type{<:Contingency}, v) = (first(size(v)) > 1) && (getindex(size(v), 2) > 1)

function Distributions.cdf(d::Contingency, x::AbstractArray{T}) where {T}
    @assert length(x) === 2 "$(x) should be length 2 vector "
    contingencymatrix = contingency_matrix(d)
    P = float(eltype(contingencymatrix))
    n = first(size(contingencymatrix))
    support = collect(1:n)
    s = zero(P)

    x[1] < Base.minimum(support) && return zero(P)
    x[2] < Base.minimum(support) && return zero(P)

    if x[1] >= Base.maximum(support) && x[2] >= Base.maximum(support)
        return one(P)
    end

    stop_idx = searchsortedlast(support, x[1])
    stop_idy = searchsortedlast(support, x[2])

    if iszero(stop_idx) && !iszero(stop_idy)
        s = sum(contingencymatrix[1, 1:stop_idy])
    elseif iszero(stop_idx) && iszero(stop_idy)
        s = contingencymatrix[1, 1]
    elseif !iszero(stop_idx) && iszero(stop_idy)
        s = sum(contingencymatrix[1:stop_idx, 1])
    else
        s = sum(contingencymatrix[1:stop_idx, 1:stop_idy])
    end
    return s
end

function icdf(dist::Contingency, probability::Float64)
    @assert 0 <= probability <= 1 "probability should be between 0 and 1."
    contingencymatrix = contingency_matrix(dist)
    support = collect(1:first(size(contingencymatrix)))
    cdfmatrix = zeros(length(support), length(support))
    @inbounds for (s1, i) in zip(support, collect(1:length(support)))
        @inbounds for (s2, j) in zip(support, collect(1:length(support)))
            cdfmatrix[i, j] = cdf(dist, [s1, s2])
        end
    end
    cartesianind = findall(x -> x == Base.minimum(filter(d -> !isless(d, probability), cdfmatrix)), cdfmatrix)

    return [cartesianind[1][1], cartesianind[1][2]]
end

isproper(params::NaturalParameters{Contingency}) = true
basemeasure(::Union{<:NaturalParameters{Contingency}, <:Contingency}, x) = 1.0

function Random.rand(rng::AbstractRNG, dist::Contingency{T}) where {T}
    probvector   = vec(contingency_matrix(dist))
    sampleindex  = rand(rng, Categorical(probvector))
    cartesianind = indexin(probvector[sampleindex], contingency_matrix(dist))
    return [cartesianind[1][1], cartesianind[1][2]]
end

function Random.rand(rng::AbstractRNG, dist::Contingency{T}, size::Int64) where {T}
    container = Matrix{T}(undef, 2, size)
    return rand!(rng, dist, container)
end

function Random.rand!(rng::AbstractRNG, dist::Contingency, container::AbstractArray{T}) where {T <: Real}
    preallocated = similar(container)
    @inbounds for i in 1:size(preallocated, 2)
        temp = rand(rng, dist)
        @views container[:, i] = temp
    end
    return container
end

plus(::NaturalParameters{Contingency}, ::NaturalParameters{Contingency}) = Plus()
