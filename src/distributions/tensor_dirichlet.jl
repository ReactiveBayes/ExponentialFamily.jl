export TensorDirichlet, ContinuousTensorDistribution

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf
using Distributions

import SparseArrays: blockdiag, sparse
import FillArrays: Ones, Eye
import LoopVectorization: vmap, vmapreduce
using LinearAlgebra, Random

const ContinuousTensorDistribution = Distribution{ ArrayLikeVariate, Continuous} 

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
struct TensorDirichlet{T<:Real, N , A <: AbstractArray{T,N}} <: ContinuousTensorDistribution
    a::A
end

extract_collection(dist::TensorDirichlet) = [dist.a[:,i] for i in CartesianIndices(Base.tail(size(dist.a)))]
unpack_parameters(::Type{TensorDirichlet}, packed) = ([packed[:,i] for i in CartesianIndices(Base.tail(size(packed)))],)
BayesBase.params(::MeanParametersSpace, dist::TensorDirichlet) = (reduce(vcat,extract_collection(dist)),)
getbasemeasure(::Type{TensorDirichlet}) = (x) -> sum([x[:,i] for i in CartesianIndices(Base.tail(size(x)))])
getsufficientstatistics(::TensorDirichlet) = (x -> vmap(log, x),)


BayesBase.mean(dist::TensorDirichlet) = dist.a ./ sum(dist.a; dims = 1)
BayesBase.cov(dist::TensorDirichlet) = map(d->cov(Dirichlet(d)),extract_collection(dist))
BayesBase.var(dist::TensorDirichlet) = map(d->var(Dirichlet(d)),extract_collection(dist))
BayesBase.std(dist::TensorDirichlet) = map(d->std(Dirichlet(d)),extract_collection(dist))


BayesBase.params(dist::TensorDirichlet) = (dist.a,)

Base.size(dist::TensorDirichlet) = size(dist.a)
Base.eltype(::TensorDirichlet{T}) where {T} = T

function BayesBase.vague(::Type{<:TensorDirichlet}, dims::Int)    
    return TensorDirichlet(ones(dims,dims))
end


function BayesBase.vague(::Type{<:TensorDirichlet}, dims::Tuple) 
    return TensorDirichlet(ones(Float64,dims))      
end

function BayesBase.entropy(dist::TensorDirichlet)
    return vmapreduce(+, extract_collection(dist)) do column
        scolumn = sum(column)
        -sum((column .- one(Float64)) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) +
        sum(loggamma.(column))
    end
end

BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:Dirichlet})  = Dirichlet
BayesBase.promote_variate_type(::Type{Multivariate}, ::Type{<:TensorDirichlet})  = TensorDirichlet
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

function BayesBase.rand!(rng::AbstractRNG, dist::TensorDirichlet, container::AbstractArray{T,N}) where {T <: Real, N }
    for index in CartesianIndices(extract_collection(dist))
        rand!(rng, Dirichlet(dist.a[:,index]), @view container[:,index])
    end
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::TensorDirichlet{A}, container::AbstractArray{A,N}) where {T <: Real, N, A <: AbstractArray{T,N}}
    for i in container
        rand!(rng, dist, @view container[i])
    end
    return container
end

function BayesBase.logpdf(dist::TensorDirichlet, x::AbstractArray{T,N}) where {T <: Real, N}
    out = 0
    for i in CartesianIndices(extract_collection(dist.a))
        out =+ logpdf(Dirichlet(dist.a[:,i]), @view x[:,i])
    end
    return out
end

BayesBase.pdf(dist::TensorDirichlet, x::Array{T,N}) where {T <: Real,N}  = exp(logpdf(dist, x))

BayesBase.default_prod_rule(::Type{<:TensorDirichlet}, ::Type{<:TensorDirichlet}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::TensorDirichlet, right::TensorDirichlet)
    paramL = extract_collection(left)
    paramR = extract_collection(right)
    Ones = ones(size(left.a))
    Ones = extract_collection(TensorDirichlet(Ones))
    param =  copy(Ones)
    for i in eachindex(paramL)
        param[i] .= paramL[i] .+ paramR[i] .- Ones[i]
    end
    out = similar(left.a)
    for i in CartesianIndices(param)
        out[:,i] = param[i]
    end
    return TensorDirichlet(out)
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{TensorDirichlet}, x)
    l = size(getnaturalparameters(ef))
    values = [x[:,i] for i in CartesianIndices(Base.tail(size(x)))]
    ## The element of the array should be the a categorical distribution (an vector of postive value that sum to 1)
    ## and all catagorical distribution should have the same size than the corresponding disrichlet prior (not checked).
    return l == size(x) && all(x ->sum(x) ≈ 1, values)  && all(!any(x-> x < 0 ), values)
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{TensorDirichlet}, η, conditioner) =
    isnothing(conditioner) && length(η) > 1 && all( map(x->isproper(NaturalParametersSpace(),Dirichlet,x,), unpack_parameters(TensorDirichlet, η)))
isproper(::MeanParametersSpace, ::Type{TensorDirichlet}, θ, conditioner) =
    isnothing(conditioner) && length(θ) > 1 && all( map(x->isproper(MeanParametersSpace(),Dirichlet,x,),unpack_parameters(TensorDirichlet, θ)))
isproper(p, ::Type{TensorDirichlet}, η, conditioner) =
    isnothing(conditioner) && all(x->isproper(p,Type{Dirichlet},x),unpack_parameters(TensorDirichlet, η))


function (::MeanToNatural{TensorDirichlet})(tuple_of_θ::Tuple{Any})
    (α,) = tuple_of_θ
    out = copy(α)
    for i in CartesianIndices(Base.tail(size(α)))
        out[:,i] = α[:,i] - ones(length(α[:,i]))
    end
    return out
end

function (::NaturalToMean{TensorDirichlet})(tuple_of_η::Tuple{Any})
    (α,) = tuple_of_η
    out = copy(α)
    for i in CartesianIndices(Base.tail(size(α)))
        out[:,i] = α[:,i] + ones(length(α[:,i]))
    end
    return out
end


getlogpartition(::NaturalParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return mapreduce(x->getlogpartition(NaturalParametersSpace(),Dirichlet)(x),+,η)
    end

getgradlogpartition(::NaturalParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return map(d -> getgradlogpartition(NaturalParametersSpace(), Dirichlet)(d), η)
    end

getfisherinformation(::NaturalParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return mapreduce(d -> getfisherinformation(NaturalParametersSpace(), Dirichlet)(d),+, η)
    end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return mapreduce(x->getlogpartition(MeanParametersSpace(),Dirichlet)(x),+,η)
    end

getgradlogpartition(::MeanParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return map(d -> getgradlogpartition(MeanParametersSpace(), Dirichlet)(d), η)
    end

getfisherinformation(::MeanParametersSpace, ::Type{TensorDirichlet}) =
    (η) -> begin
        return mapreduce(d -> getfisherinformation(MeanParametersSpace(), Dirichlet)(d),+, η)
    end

