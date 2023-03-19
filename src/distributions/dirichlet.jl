export Dirichlet

import Distributions: Dirichlet
import SpecialFunctions: digamma, loggamma

vague(::Type{<:Dirichlet}, dims::Int) = Dirichlet(ones(dims))

prod_analytical_rule(::Type{<:Dirichlet}, ::Type{<:Dirichlet}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Dirichlet, right::Dirichlet)
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

function lognormalizer(exponentialfamily::ExponentialFamilyDistribution{Dirichlet})
    η = getnaturalparameters(exponentialfamily)
    firstterm = mapreduce(x -> loggamma(x + 1), +, η)
    secondterm = loggamma(sum(η .+ 1))
    return firstterm - secondterm
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Dirichlet})
    getnaturalparameters(exponentialfamily)
    return Dirichlet(getnaturalparameters(exponentialfamily) .+ 1)
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Dirichlet)
    ExponentialFamilyDistribution(Dirichlet, probvec(dist) .- 1)
end

isproper(exponentialfamily::ExponentialFamilyDistribution{<:Dirichlet}) = all(isless.(-1, getnaturalparameters(exponentialfamily)))

check_valid_natural(::Type{<:Dirichlet}, params) = (length(params) > 1)
basemeasure(::Union{<:ExponentialFamilyDistribution{Dirichlet}, <:Dirichlet}, x) = 1.0
plus(::ExponentialFamilyDistribution{Dirichlet}, ::ExponentialFamilyDistribution{Dirichlet}) = Plus()
