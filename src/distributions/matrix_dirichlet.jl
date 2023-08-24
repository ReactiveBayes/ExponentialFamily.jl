export MatrixDirichlet

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf

import SparseArrays: blockdiag, sparse
import FillArrays: Ones, Eye
import LoopVectorization: vmap, vmapreduce
using LinearAlgebra, Random

"""
    MatrixDirichlet{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution

A matrix-valued MatrixDirichlet distribution, where `T` is the element type of the matrix `A`.
The `a` field stores the matrix parameter of the distribution.

# Fields
- `a::A`: The matrix parameter of the MatrixDirichlet distribution.
"""
struct MatrixDirichlet{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    a::A
end

Distributions.mean(dist::MatrixDirichlet) = dist.a ./ sum(dist.a; dims = 1)
function Distributions.cov(dist::MatrixDirichlet)

    matrices = vmap(
        d -> sparse(cov(Dirichlet(convert(Vector,d)))),
        eachcol(dist.a)
    )

    return vmapreduce(identity, blockdiag, matrices)
end

Distributions.var(dist::MatrixDirichlet) = diag(cov(dist))
Distributions.std(dist::MatrixDirichlet) = vmap(sqrt, var(dist))

params(dist::MatrixDirichlet) = (dist.a, )
size(dist::MatrixDirichlet)   = size(dist.a)

Base.eltype(::MatrixDirichlet{T}) where {T} = T

vague(::Type{<:MatrixDirichlet}, dims::Int)              = MatrixDirichlet(ones(dims, dims))
vague(::Type{<:MatrixDirichlet}, dims1::Int, dims2::Int) = MatrixDirichlet(ones(dims1, dims2))
vague(::Type{<:MatrixDirichlet}, dims::Tuple)            = MatrixDirichlet(ones(dims))

function Distributions.entropy(dist::MatrixDirichlet)
    return vmapreduce(+, eachcol(dist.a)) do column
        scolumn = sum(column)
        -sum((column .- one(Float64)) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) +
        sum(loggamma.(column))
    end
end

promote_variate_type(::Type{Multivariate}, ::Type{<:Dirichlet})  = Dirichlet
promote_variate_type(::Type{Matrixvariate}, ::Type{<:Dirichlet}) = MatrixDirichlet

promote_variate_type(::Type{Multivariate}, ::Type{<:MatrixDirichlet})  = Dirichlet
promote_variate_type(::Type{Matrixvariate}, ::Type{<:MatrixDirichlet}) = MatrixDirichlet

function Random.rand(rng::AbstractRNG, dist::MatrixDirichlet{T}) where {T}
    container = similar(dist.a)
    return rand!(rng, dist, container)
end

function Random.rand(rng::AbstractRNG, dist::MatrixDirichlet{T}, nsamples::Int64) where {T}
    container = Vector{Matrix{T}}(undef, nsamples)
    @inbounds for i in eachindex(container)
        container[i] = Matrix{T}(undef, size(dist))
        rand!(rng, dist, container[i])
    end
    return container
end

function Random.rand!(rng::AbstractRNG, dist::MatrixDirichlet, container::AbstractMatrix{T}) where {T <: Real}
    samples = vmap(d -> rand(rng, Dirichlet(convert(Vector,d))), eachcol(dist.a))
    @views for row in 1:isqrt(length(container))
        b = container[:, row]
        b[:] .= samples[row]
    end

    return container 
end

function Random.rand!(rng::AbstractRNG, dist::MatrixDirichlet, container::AbstractVector{T}) where {T <: AbstractMatrix}
    for i in eachindex(container)
        rand!(rng, dist, container[i])
    end
    return container
end

function Distributions.logpdf(dist::MatrixDirichlet, x::Matrix)
    return vmapreduce(
        d -> logpdf(Dirichlet(convert(Vector,d[1])),convert(Vector,d[2])),
        +,
        zip(eachcol(dist.a),eachcol(x))
    )
end

Distributions.pdf(dist::MatrixDirichlet, x::Matrix) = exp(logpdf(dist, x))

mean(::typeof(log), dist::MatrixDirichlet) = digamma.(dist.a) .- digamma.(sum(dist.a; dims = 1))

default_prod_rule(::Type{<:MatrixDirichlet}, ::Type{<:MatrixDirichlet}) = ClosedProd()

function Base.prod(::ClosedProd, left::MatrixDirichlet, right::MatrixDirichlet)
    T = promote_samplefloattype(left, right)
    return MatrixDirichlet(left.a + right.a - Ones{T}(size(left.a)))
end

function insupport(ef::ExponentialFamilyDistribution{MatrixDirichlet}, x) 
    l = length(getnaturalparameters(ef))
    ## The columns of x should be normalized. all(≈(1), sum(eachrow(x))) is a convenient way of doing that
    ## because eachrow(x) will return row slices and sum will take the sum of the row slices along the first dimension
    return l == length(x) && !any(x -> x < zero(x), x) && all(≈(1), sum(eachrow(x)))
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{MatrixDirichlet}, η, conditioner) = isnothing(conditioner) && length(η) > 1  && typeof(isqrt(length(η))) <: Integer && all(isless.(-1, η)) && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{MatrixDirichlet}, θ, conditioner) = isnothing(conditioner) && length(θ) > 1&& typeof(isqrt(length(θ))) <: Integer && all(>(0), θ) && all(!isinf, θ) && all(!isnan, θ)

function (::MeanToNatural{MatrixDirichlet})(tuple_of_θ::Tuple{Any})
    (α, ) = tuple_of_θ
    return (α - Ones{Float64}(size(α)) , )
end

function (::NaturalToMean{MatrixDirichlet})(tuple_of_η::Tuple{Any})
    (η, ) = tuple_of_η
    return (η + Ones{Float64}(size(η)), )
end

function unpack_parameters(::Type{MatrixDirichlet}, packed) 
    n = length(packed)
    return (reshape(view(packed, 1:n), isqrt(n), isqrt(n)), )
end

isbasemeasureconstant(::Type{MatrixDirichlet}) = ConstantBaseMeasure()

getbasemeasure(::Type{MatrixDirichlet}) = (x) -> one(Float64)
getsufficientstatistics(::Type{MatrixDirichlet}) = (x -> vmap(log,x), )

getlogpartition(::NaturalParametersSpace, ::Type{MatrixDirichlet}) = (η) -> begin
    (η1, ) = unpack_parameters(MatrixDirichlet, η)
    return   vmapreduce(
        d -> getlogpartition(NaturalParametersSpace(), Dirichlet)(convert(Vector,d)),
        +,
        eachcol(η1)
    )
end

getfisherinformation(::NaturalParametersSpace, ::Type{MatrixDirichlet}) = (η) -> begin
    (η1, ) = unpack_parameters(MatrixDirichlet, η)
    ones = Ones{Float64}(size(η1))
    ηp1 = η1 + ones

    matrices = map(d -> sparse(Diagonal(d[2]) - d[1] * ones),
        Iterators.zip(map(d -> trigamma(d), sum(ηp1, dims = 1)), map(d -> trigamma.(d), eachcol(ηp1))))

    return vmapreduce(identity ,blockdiag, matrices)
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{MatrixDirichlet}) = (θ) -> begin
    (α, ) = unpack_parameters(MatrixDirichlet, θ)
    return   vmapreduce(
        d -> getlogpartition(MeanParametersSpace(), Dirichlet)(convert(Vector,d)),
        +,
        eachcol(α)
    )
end

getfisherinformation(::MeanParametersSpace, ::Type{MatrixDirichlet}) = (θ) -> begin
    (α,  ) = unpack_parameters(MatrixDirichlet, θ)
    matrices = map(d -> sparse(Diagonal(d[2]) - d[1] * Ones{Float64}(size(α))),
        Iterators.zip(map(d -> trigamma(d), sum(α, dims = 1)), map(d -> trigamma.(d), eachcol(α))))

    return vmapreduce(identity ,blockdiag, matrices)
end



































# function pack_naturalparameters(distribution::MatrixDirichlet)
#     return vec(distribution.a) - Ones{Float64}(vectorized_length(distribution))
# end
# function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:MatrixDirichlet})
#     vectorized = getnaturalparameters(ef)
#     len = length(vectorized)
#     Ssize = isqrt(len)
#     return (reshape(view(vectorized, 1:len), Ssize, Ssize),)
# end

# ##TODO: this code needs to be optimized
# logpartition(exponentialfamily::ExponentialFamilyDistribution{MatrixDirichlet}) =
#     vmapreduce(
#         d -> logpartition(ExponentialFamilyDistribution(MatrixDirichlet, convert(Vector, d))),
#         +,
#         eachcol(first(unpack_naturalparameters(exponentialfamily)))
#     )

# Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{MatrixDirichlet}) =
#     MatrixDirichlet(first(unpack_naturalparameters(exponentialfamily)) .+ one(Float64))

# function Base.convert(::Type{ExponentialFamilyDistribution}, dist::MatrixDirichlet)
#     ExponentialFamilyDistribution(MatrixDirichlet, pack_naturalparameters(dist))
# end

# isproper(exponentialfamily::ExponentialFamilyDistribution{<:MatrixDirichlet}) =
#     all(isless.(-1, getnaturalparameters(exponentialfamily)))

# check_valid_natural(::Type{<:MatrixDirichlet}, params) = (typeof(params) <: Vector)

# basemeasure(::Union{<:ExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet}) = one(Float64)
# function basemeasure(
#     ::Union{<:ExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet},
#     x::Matrix{T}
# ) where {T}
#     return one(eltype(x))
# end
# sufficientstatistics(ef::Union{<:ExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet}) =
#     x -> sufficientstatistics(ef, x)
# function sufficientstatistics(
#     ::Union{<:ExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet},
#     x::Matrix{T}
# ) where {T}
#     return vec(vmap(d -> log(d), x))
# end

# # #this works  50 allocations
# # function fisherinformation(ef::ExponentialFamilyDistribution{MatrixDirichlet})
# #     ηp1 = unpack_naturalparameters(ef) .+ 1
# #     ηvect = collect(Vector, eachcol(ηp1))
# #     n = length(ηvect)
# #     ηvect0 = sum(ηvect)
# #     matrices = [@inbounds sparse(diagm(trigamma.(ηvect[i])) - Ones{Float64}(n, n) * trigamma(ηvect0[i])) for i in 1:n]
# #     return blockdiag(Tuple(matrices)...)
# # end

# ## this works 48 allocations
# # function fisherinformation(ef::ExponentialFamilyDistribution{MatrixDirichlet})
# #     η = unpack_naturalparameters(ef)
# #     ones = Ones{Float64}(size(η))
# #     ηvect0_trigammas = map(d -> ones*d, map(d ->trigamma(d), sum(η+ones,dims=1)))
# #     trigammas = map(d -> diagm(d), map(d -> trigamma.(d), eachcol(η+ones)))

# #     blockdiag(Tuple(map(d -> sparse(d[2] - d[1]) , Iterators.zip(Tuple(trigammas),Tuple(ηvect0_trigammas))))...)
# # end

# ## this works 36 allocations
# function fisherinformation(ef::ExponentialFamilyDistribution{MatrixDirichlet})
#     (η,) = unpack_naturalparameters(ef)
#     ones = Ones{Float64}(size(η))
#     ηp1 = η + ones

#     matrices = map(d -> sparse(Diagonal(d[2]) - d[1] * ones),
#         Iterators.zip(map(d -> trigamma(d), sum(ηp1, dims = 1)), map(d -> trigamma.(d), eachcol(ηp1))))

#     return blockdiag(Tuple(matrices)...)
# end

# function fisherinformation(dist::MatrixDirichlet)
#     ηp1 = dist.a

#     matrices = map(d -> sparse(Diagonal(d[2]) - d[1] * Ones{Float64}(size(ηp1))),
#         Iterators.zip(map(d -> trigamma(d), sum(ηp1, dims = 1)), map(d -> trigamma.(d), eachcol(ηp1))))

#     return blockdiag(Tuple(matrices)...)
# end
