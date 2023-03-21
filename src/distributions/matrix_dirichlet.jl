export MatrixDirichlet

import SpecialFunctions: digamma, loggamma
import Base: eltype
import Distributions: pdf, logpdf

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
        -sum((column .- 1.0) .* (digamma.(column) .- digamma.(scolumn))) - loggamma(scolumn) + sum(loggamma.(column))
    end
end

function Distributions.logpdf(dist::MatrixDirichlet, x::Matrix)
    η = Base.convert(KnownExponentialFamilyDistribution, dist)
    return -logpartition(η) + tr(getnaturalparameters(η)' * log.(x))
end

Distributions.pdf(dist::MatrixDirichlet, x::Matrix) = exp(logpdf(dist, x))

mean(::typeof(log), dist::MatrixDirichlet) = digamma.(dist.a) .- digamma.(sum(dist.a; dims = 1))

prod_analytical_rule(::Type{<:MatrixDirichlet}, ::Type{<:MatrixDirichlet}) = ClosedProd()

function Base.prod(::ClosedProd, left::MatrixDirichlet, right::MatrixDirichlet)
    T = promote_samplefloattype(left, right)
    return MatrixDirichlet(left.a + right.a .- one(T))
end

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{MatrixDirichlet}) =
    mapreduce(d -> logpartition(KnownExponentialFamilyDistribution(Dirichlet, d)), +, eachrow(getnaturalparameters(exponentialfamily)))

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{MatrixDirichlet})
    getnaturalparameters(exponentialfamily)
    return MatrixDirichlet(getnaturalparameters(exponentialfamily) .+ 1)
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::MatrixDirichlet)
    KnownExponentialFamilyDistribution(MatrixDirichlet, dist.a .- 1)
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{<:MatrixDirichlet}) = all(isless.(-1, getnaturalparameters(exponentialfamily)))

check_valid_natural(::Type{<:MatrixDirichlet}, params) = (typeof(params) <: Matrix)

basemeasure(::Union{<:KnownExponentialFamilyDistribution{MatrixDirichlet}, <:MatrixDirichlet}, x) = 1.0
