export MatrixDirichlet

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf

import SparseArrays: blockdiag, sparse
import FillArrays: Ones, Eye
import LoopVectorization: vmap, vmapreduce
using LinearAlgebra

"""
MatrixDirichlet{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution

A matrix-valued Dirichlet distribution, where `T` is the element type of the matrix `A`.
The `a` field stores the matrix parameter of the distribution.

# Fields
- `a::A`: The matrix parameter of the Dirichlet distribution.
"""
struct MatrixDirichlet{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    a::A
end

Distributions.mean(dist::MatrixDirichlet) = dist.a ./ sum(dist.a; dims = 1)

Base.eltype(::MatrixDirichlet{T}) where {T} = T

vague(::Type{<:MatrixDirichlet}, dims::Int)              = MatrixDirichlet(ones(dims, dims))
vague(::Type{<:MatrixDirichlet}, dims1::Int, dims2::Int) = MatrixDirichlet(ones(dims1, dims2))
vague(::Type{<:MatrixDirichlet}, dims::Tuple)            = MatrixDirichlet(ones(dims))
vectorized_length(dist::MatrixDirichlet)                 = length(dist.a)

function Distributions.entropy(dist::MatrixDirichlet)
    return mapreduce(+, eachcol(dist.a)) do column
        scolumn = sum(column)
        -sum((column .- one(Float64)) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) +
        sum(loggamma.(column))
    end
end

function Distributions.logpdf(dist::MatrixDirichlet, x::Matrix)
    ef = Base.convert(ExponentialFamilyDistribution, dist)
    return -logpartition(ef) + tr(first(unpack_naturalparameters(ef))' * log.(x))
end

Distributions.pdf(dist::MatrixDirichlet, x::Matrix) = exp(logpdf(dist, x))

mean(::typeof(log), dist::MatrixDirichlet) = digamma.(dist.a) .- digamma.(sum(dist.a; dims = 1))

closed_prod_rule(::Type{<:MatrixDirichlet}, ::Type{<:MatrixDirichlet}) = ClosedProd()

function Base.prod(::ClosedProd, left::MatrixDirichlet, right::MatrixDirichlet)
    T = promote_samplefloattype(left, right)
    return MatrixDirichlet(left.a + right.a - Ones{T}(size(left.a)))
end

function pack_naturalparameters(distribution::MatrixDirichlet)
    return vec(distribution.a) - Ones{Float64}(vectorized_length(distribution))
end
function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:MatrixDirichlet})
    vectorized = getnaturalparameters(ef)
    len = length(vectorized)
    Ssize = isqrt(len)
    return (reshape(view(vectorized, 1:len), Ssize, Ssize),)
end

##TODO: this code needs to be optimized
logpartition(exponentialfamily::ExponentialFamilyDistribution{MatrixDirichlet}) =
    vmapreduce(
        d -> logpartition(ExponentialFamilyDistribution(Dirichlet, convert(Vector, d))),
        +,
        eachcol(first(unpack_naturalparameters(exponentialfamily)))
    )

Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{MatrixDirichlet}) =
    MatrixDirichlet(first(unpack_naturalparameters(exponentialfamily)) .+ one(Float64))

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::MatrixDirichlet)
    ExponentialFamilyDistribution(MatrixDirichlet, pack_naturalparameters(dist))
end

isproper(exponentialfamily::ExponentialFamilyDistribution{<:MatrixDirichlet}) =
    all(isless.(-1, getnaturalparameters(exponentialfamily)))

check_valid_natural(::Type{<:MatrixDirichlet}, params) = (typeof(params) <: Vector)

basemeasure(::Union{<:ExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet}) = one(Float64)
function basemeasure(
    ::Union{<:ExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet},
    x::Matrix{T}
) where {T}
    return one(eltype(x))
end
sufficientstatistics(ef::Union{<:ExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet}) =
    x -> sufficientstatistics(ef, x)
function sufficientstatistics(
    ::Union{<:ExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet},
    x::Matrix{T}
) where {T}
    return vec(vmap(d -> log(d), x))
end

# #this works  50 allocations
# function fisherinformation(ef::ExponentialFamilyDistribution{MatrixDirichlet})
#     ηp1 = unpack_naturalparameters(ef) .+ 1
#     ηvect = collect(Vector, eachcol(ηp1))
#     n = length(ηvect)
#     ηvect0 = sum(ηvect)
#     matrices = [@inbounds sparse(diagm(trigamma.(ηvect[i])) - Ones{Float64}(n, n) * trigamma(ηvect0[i])) for i in 1:n]
#     return blockdiag(Tuple(matrices)...)
# end

## this works 48 allocations
# function fisherinformation(ef::ExponentialFamilyDistribution{MatrixDirichlet})
#     η = unpack_naturalparameters(ef)
#     ones = Ones{Float64}(size(η))
#     ηvect0_trigammas = map(d -> ones*d, map(d ->trigamma(d), sum(η+ones,dims=1)))
#     trigammas = map(d -> diagm(d), map(d -> trigamma.(d), eachcol(η+ones)))

#     blockdiag(Tuple(map(d -> sparse(d[2] - d[1]) , Iterators.zip(Tuple(trigammas),Tuple(ηvect0_trigammas))))...)
# end

## this works 36 allocations
function fisherinformation(ef::ExponentialFamilyDistribution{MatrixDirichlet})
    (η,) = unpack_naturalparameters(ef)
    ones = Ones{Float64}(size(η))
    ηp1 = η + ones

    matrices = map(d -> sparse(Diagonal(d[2]) - d[1] * ones),
        Iterators.zip(map(d -> trigamma(d), sum(ηp1, dims = 1)), map(d -> trigamma.(d), eachcol(ηp1))))

    return blockdiag(Tuple(matrices)...)
end

function fisherinformation(dist::MatrixDirichlet)
    ηp1 = dist.a

    matrices = map(d -> sparse(Diagonal(d[2]) - d[1] * Ones{Float64}(size(ηp1))),
        Iterators.zip(map(d -> trigamma(d), sum(ηp1, dims = 1)), map(d -> trigamma.(d), eachcol(ηp1))))

    return blockdiag(Tuple(matrices)...)
end
