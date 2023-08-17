export Dirichlet

import Distributions: Dirichlet, std
import SpecialFunctions: digamma, loggamma, trigamma
using FillArrays
using LoopVectorization
using StaticArrays
using LinearAlgebra
using LogExpFunctions

vague(::Type{<:Dirichlet}, dims::Int) = Dirichlet(ones(dims))

default_prod_rule(::Type{<:Dirichlet}, ::Type{<:Dirichlet}) = ClosedProd()

function Base.prod(::ClosedProd, left::Dirichlet, right::Dirichlet)
    mvec = probvec(left) .+ probvec(right)
    mvec = mvec .- one(eltype(mvec))
    return Dirichlet(mvec)
end

probvec(dist::Dirichlet) = params(dist)[1]
std(dist::Dirichlet) = vmap(sqrt, var(dist))
mean(::typeof(log), dist::Dirichlet)      = digamma.(probvec(dist)) .- digamma(sum(probvec(dist)))
mean(::typeof(clamplog), dist::Dirichlet) = digamma.((clamp(p, tiny, typemax(p)) for p in probvec(dist))) .- digamma(sum(probvec(dist)))

# promote_variate_type(::Type{Multivariate}, ::Type{<:Dirichlet})  = Dirichlet
# promote_variate_type(::Type{Matrixvariate}, ::Type{<:Dirichlet}) = MatrixDirichlet

# promote_variate_type(::Type{Multivariate}, ::Type{<:MatrixDirichlet})  = Dirichlet
# promote_variate_type(::Type{Matrixvariate}, ::Type{<:MatrixDirichlet}) = MatrixDirichlet

function compute_logscale(new_dist::Dirichlet, left_dist::Dirichlet, right_dist::Dirichlet)
    return logmvbeta(probvec(new_dist)) - logmvbeta(probvec(left_dist)) - logmvbeta(probvec(right_dist))
end

function insupport(ef::ExponentialFamilyDistribution{Dirichlet}, x) 
    l = length(getnaturalparameters(ef))
    return l == length(x) && !any(x -> x < zero(x), x) && sum(x) ≈ 1
end
# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Dirichlet}, η, conditioner) = isnothing(conditioner) && length(η) > 1 &&  all(isless.(-1, η)) && all(!isinf, η)
isproper(::MeanParametersSpace, ::Type{Dirichlet}, θ, conditioner) = isnothing(conditioner) && length(θ) > 1 && all(>(0), θ) && all(!isinf, θ)

function (::MeanToNatural{Dirichlet})(tuple_of_θ::Tuple{Any})
    (α, ) = tuple_of_θ
    return (α - Ones{Float64}(length(α)) , )
end

function (::NaturalToMean{Dirichlet})(tuple_of_η::Tuple{Any})
    (η, ) = tuple_of_η
    return (η + Ones{Float64}(length(η)), )
end

unpack_parameters(::Type{Dirichlet}, packed) = (packed, )

isbasemeasureconstant(::Type{Dirichlet}) = ConstantBaseMeasure()

getbasemeasure(::Type{Dirichlet}) = (x) -> 1.0
getsufficientstatistics(::Type{Dirichlet}) = (x -> vmap(log,x), )

getlogpartition(::NaturalParametersSpace, ::Type{Dirichlet}) = (η) -> begin
    (η1, ) = unpack_parameters(Dirichlet, η)
    firstterm = mapreduce(x -> loggamma(x + 1), +, η1)
    secondterm = loggamma(sum(η1) + length(η1))
    return firstterm - secondterm
end

getfisherinformation(::NaturalParametersSpace, ::Type{Dirichlet}) = (η) -> begin
    (η1, ) = unpack_parameters(Dirichlet, η)
    n = length(η1)
    return Diagonal(map(d -> trigamma(d + 1), η1)) - Ones{Float64}(n, n) * trigamma(sum(η1) + n)
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Dirichlet}) = (θ) -> begin
    (α, ) = unpack_parameters(Dirichlet, θ)
    firstterm = mapreduce(x -> loggamma(x), +, α)
    secondterm = loggamma(sum(α))
    return firstterm - secondterm
end

getfisherinformation(::MeanParametersSpace, ::Type{Dirichlet}) = (θ) -> begin
    (α,  ) = unpack_parameters(Dirichlet, θ)
    n = length(α)
    return Diagonal(map(d -> trigamma(d), α)) - Ones{Float64}(n, n) * trigamma(sum(α))
end
