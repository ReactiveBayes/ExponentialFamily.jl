export Dirichlet

import Distributions: Dirichlet
import SpecialFunctions: digamma, loggamma, trigamma
using FillArrays
using LoopVectorization
using StaticArrays
using LinearAlgebra
using LogExpFunctions


vague(::Type{<:Dirichlet}, dims::Int) = Dirichlet(ones(dims))

closed_prod_rule(::Type{<:Dirichlet}, ::Type{<:Dirichlet}) = ClosedProd()

function Base.prod(::ClosedProd, left::Dirichlet, right::Dirichlet)
    mvec = probvec(left) .+ probvec(right)
    mvec = mvec .- one(eltype(mvec))
    return Dirichlet(mvec)
end

probvec(dist::Dirichlet) = params(dist)[1]

mean(::typeof(log), dist::Dirichlet)      = digamma.(probvec(dist)) .- digamma(sum(probvec(dist)))
mean(::typeof(clamplog), dist::Dirichlet) = digamma.((clamp(p, tiny, typemax(p)) for p in probvec(dist))) .- digamma(sum(probvec(dist)))

promote_variate_type(::Type{Multivariate}, ::Type{<:Dirichlet})  = Dirichlet
promote_variate_type(::Type{Matrixvariate}, ::Type{<:Dirichlet}) = MatrixDirichlet

promote_variate_type(::Type{Multivariate}, ::Type{<:MatrixDirichlet})  = Dirichlet
promote_variate_type(::Type{Matrixvariate}, ::Type{<:MatrixDirichlet}) = MatrixDirichlet

function compute_logscale(new_dist::Dirichlet, left_dist::Dirichlet, right_dist::Dirichlet)
    return logmvbeta(probvec(new_dist)) - logmvbeta(probvec(left_dist)) - logmvbeta(probvec(right_dist))
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Dirichlet})
    η = getnaturalparameters(exponentialfamily)
    firstterm = mapreduce(x -> loggamma(x + 1), +, η)
    secondterm = loggamma(sum(η)+ length(η))
    return firstterm - secondterm
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Dirichlet})
    η = getnaturalparameters(exponentialfamily)
    return Dirichlet(η + Ones{Float64}(length(η)))
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Dirichlet)
    ExponentialFamilyDistribution(Dirichlet, probvec(dist) - Ones{Float64}(length(probvec(dist))))
end

isproper(exponentialfamily::ExponentialFamilyDistribution{<:Dirichlet}) =
    all(isless.(-1, getnaturalparameters(exponentialfamily)))

check_valid_natural(::Type{<:Dirichlet}, params) = (length(params) > one(Int64))

function insupport(ef::ExponentialFamilyDistribution{Dirichlet, P, C, Safe}, x) where {P, C}
    l = length(getnaturalparameters(ef))
    return l == length(x) && !any(x -> x < zero(x), x) && sum(x) ≈ 1
end

function basemeasure(ef::ExponentialFamilyDistribution{Dirichlet}, x)
    @assert insupport(ef, x) "$(x) is not in support of Dirichlet"
    return one(eltype(x))
end

## has one allocation
function sufficientstatistics(ef::ExponentialFamilyDistribution{Dirichlet}, x)
    @assert insupport(ef, x) "$(x) is not in support of Dirichlet"
    return vmap(d -> log(d), x)
end

function fisherinformation(dist::Dirichlet)
    α  = probvec(dist)
    n  = length(α)
    return Diagonal(map(d->trigamma(d),α)) - Ones{Float64}(n, n) * trigamma(sum(α))
end

function fisherinformation(ef::ExponentialFamilyDistribution{Dirichlet})
    η = getnaturalparameters(ef)
    n = length(η)
    return Diagonal(map(d -> trigamma(d + 1), η)) - Ones{Float64}(n, n) * trigamma(sum(η) + n)
end
