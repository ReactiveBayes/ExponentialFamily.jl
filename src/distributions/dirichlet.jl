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

# probvec is not normalised
mean(::typeof(log), dist::Dirichlet)      = digamma.(probvec(dist)) .- digamma(sum(probvec(dist)))
mean(::typeof(clamplog), dist::Dirichlet) = digamma.((clamp(p, tiny, typemax(p)) for p in probvec(dist))) .- digamma(sum(probvec(dist)))

# Variate forms promotion

promote_variate_type(::Type{Multivariate}, ::Type{<:Dirichlet})  = Dirichlet
promote_variate_type(::Type{Matrixvariate}, ::Type{<:Dirichlet}) = MatrixDirichlet

promote_variate_type(::Type{Multivariate}, ::Type{<:MatrixDirichlet})  = Dirichlet
promote_variate_type(::Type{Matrixvariate}, ::Type{<:MatrixDirichlet}) = MatrixDirichlet

function compute_logscale(new_dist::Dirichlet, left_dist::Dirichlet, right_dist::Dirichlet)
    return logmvbeta(probvec(new_dist)) - logmvbeta(probvec(left_dist)) - logmvbeta(probvec(right_dist))
end

function lognormalizer(params::NaturalParameters{Dirichlet})
    η = get_params(params)
    firstterm = mapreduce(x -> loggamma(x + 1), +, η)
    secondterm = loggamma(sum(η .+ 1))
    return firstterm - secondterm
end

function Base.convert(::Type{Distribution}, params::NaturalParameters{Dirichlet})
    get_params(params)
    return Dirichlet(get_params(params) .+ 1)
end

function Base.convert(::Type{NaturalParameters}, dist::Dirichlet)
    NaturalParameters(Dirichlet, probvec(dist) .- 1)
end

isproper(params::NaturalParameters{<:Dirichlet}) = all(isless.(-1, get_params(params)))

check_valid_natural(::Type{<:Dirichlet}, params) = (length(params) > 1)
## due to alpha-1 parameterization
basemeasure(::Union{<:NaturalParameters{Dirichlet}, <:Dirichlet}, x) = 1.0
plus(::NaturalParameters{Dirichlet}, ::NaturalParameters{Dirichlet}) = Plus()