export MatrixDirichlet

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf

using BlockDiagonals

struct MatrixDirichlet{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    a::A
end

Distributions.mean(dist::MatrixDirichlet) = dist.a ./ sum(dist.a; dims = 1)

Base.eltype(::MatrixDirichlet{T}) where {T} = T

vague(::Type{<:MatrixDirichlet}, dims::Int)              = MatrixDirichlet(ones(dims, dims))
vague(::Type{<:MatrixDirichlet}, dims1::Int, dims2::Int) = MatrixDirichlet(ones(dims1, dims2))
vague(::Type{<:MatrixDirichlet}, dims::Tuple)            = MatrixDirichlet(ones(dims))

function Distributions.entropy(dist::MatrixDirichlet)
    return mapreduce(+, eachcol(dist.a)) do column
        scolumn = sum(column)
        -sum((column .- one(Float64)) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) +
        sum(loggamma.(column))
    end
end

function Distributions.logpdf(dist::MatrixDirichlet, x::Matrix)
    η = Base.convert(KnownExponentialFamilyDistribution, dist)
    return -logpartition(η) + tr(getnaturalparameters(η)' * log.(x))
end

Distributions.pdf(dist::MatrixDirichlet, x::Matrix) = exp(logpdf(dist, x))

mean(::typeof(log), dist::MatrixDirichlet) = digamma.(dist.a) .- digamma.(sum(dist.a; dims = 1))

closed_prod_rule(::Type{<:MatrixDirichlet}, ::Type{<:MatrixDirichlet}) = ClosedProd()

function Base.prod(::ClosedProd, left::MatrixDirichlet, right::MatrixDirichlet)
    T = promote_samplefloattype(left, right)
    return MatrixDirichlet(left.a + right.a .- one(T))
end

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{MatrixDirichlet}) =
    mapreduce(
        d -> logpartition(KnownExponentialFamilyDistribution(Dirichlet, d)),
        +,
        eachcol(getnaturalparameters(exponentialfamily))
    )

Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{MatrixDirichlet}) =
    MatrixDirichlet(getnaturalparameters(exponentialfamily) .+ one(Float64))

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::MatrixDirichlet)
    KnownExponentialFamilyDistribution(MatrixDirichlet, dist.a .- one(Float64))
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{<:MatrixDirichlet}) =
    all(isless.(-1, getnaturalparameters(exponentialfamily)))

check_valid_natural(::Type{<:MatrixDirichlet}, params) = (typeof(params) <: Matrix)

function basemeasure(
    ::Union{<:KnownExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet},
    x::Matrix{T}
) where {T}
    return 1.0
end

function sufficientstatistics(
    ::Union{<:KnownExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet},
    x::Matrix{T}
) where {T}
    return log.(x)
end

function fisherinformation(ef::KnownExponentialFamilyDistribution{MatrixDirichlet})
    ηp1 = getnaturalparameters(ef) .+ 1

    ηvect = collect(Vector, eachcol(ηp1))
    n = length(ηvect)

    ηvect0 = sum.(ηvect)

    pre_diag = [diagm(trigamma.(ηvect[i])) for i in 1:n]
    fi_pre = [ones(n, n) * trigamma(ηvect0[i]) for i in 1:n]

    return BlockDiagonal(pre_diag - fi_pre)
end

function fisherinformation(dist::MatrixDirichlet)
    ηp1 = dist.a

    ηvect = collect(Vector, eachcol(ηp1))
    n = length(ηvect)

    ηvect0 = sum.(ηvect)

    pre_diag = [diagm(trigamma.(ηvect[i])) for i in 1:n]
    fi_pre = [ones(n, n) * trigamma(ηvect0[i]) for i in 1:n]

    return BlockDiagonal(pre_diag - fi_pre)
end
