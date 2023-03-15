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
    η = Base.convert(NaturalParameters, dist)
    return -lognormalizer(η) + tr(get_params(η)' * log.(x))
end

Distributions.pdf(dist::MatrixDirichlet, x::Matrix) = exp(logpdf(dist, x))

mean(::typeof(log), dist::MatrixDirichlet) = digamma.(dist.a) .- digamma.(sum(dist.a; dims = 1))

prod_analytical_rule(::Type{<:MatrixDirichlet}, ::Type{<:MatrixDirichlet}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::MatrixDirichlet, right::MatrixDirichlet)
    T = promote_samplefloattype(left, right)
    return MatrixDirichlet(left.a + right.a .- one(T))
end

lognormalizer(params::NaturalParameters{MatrixDirichlet}) =
    mapreduce(d -> lognormalizer(NaturalParameters(Dirichlet, d)), +, eachrow(get_params(params)))

function Base.convert(::Type{Distribution}, params::NaturalParameters{MatrixDirichlet})
    get_params(params)
    return MatrixDirichlet(get_params(params) .+ 1)
end

function Base.convert(::Type{NaturalParameters}, dist::MatrixDirichlet)
    NaturalParameters(MatrixDirichlet, dist.a .- 1)
end

isproper(params::NaturalParameters{<:MatrixDirichlet}) = all(isless.(-1, get_params(params)))

check_valid_natural(::Type{<:MatrixDirichlet}, params) = (typeof(params) <: Matrix)

basemeasure(::Union{<:NaturalParameters{MatrixDirichlet}, <:MatrixDirichlet}, x) = 1.0
plus(::NaturalParameters{MatrixDirichlet}, ::NaturalParameters{MatrixDirichlet}) = Plus()
