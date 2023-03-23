export Dirichlet

import Distributions: Dirichlet
import SpecialFunctions: digamma, loggamma

vague(::Type{<:Dirichlet}, dims::Int) = Dirichlet(ones(dims))

prod_closed_rule(::Type{<:Dirichlet}, ::Type{<:Dirichlet}) = ClosedProd()

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

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Dirichlet})
    η = getnaturalparameters(exponentialfamily)
    firstterm = mapreduce(x -> loggamma(x + 1), +, η)
    secondterm = loggamma(sum(η .+ 1))
    return firstterm - secondterm
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Dirichlet})
    getnaturalparameters(exponentialfamily)
    return Dirichlet(getnaturalparameters(exponentialfamily) .+ one(Float64))
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Dirichlet)
    KnownExponentialFamilyDistribution(Dirichlet, probvec(dist) .- one(Float64))
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{<:Dirichlet}) =
    all(isless.(-1, getnaturalparameters(exponentialfamily)))

check_valid_natural(::Type{<:Dirichlet}, params) = (length(params) > one(Int64))
basemeasure(::Union{<:KnownExponentialFamilyDistribution{Dirichlet}, <:Dirichlet}, x) = 1.0