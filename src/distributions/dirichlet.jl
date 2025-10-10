export Dirichlet

import Distributions: Dirichlet, std
import SpecialFunctions: digamma, loggamma, trigamma
import Base.Broadcast: BroadcastFunction

using FillArrays
using StaticArrays
using LinearAlgebra
using LogExpFunctions

BayesBase.vague(::Type{<:Dirichlet}, dims::Int) = Dirichlet(ones(dims))
BayesBase.default_prod_rule(::Type{<:Dirichlet}, ::Type{<:Dirichlet}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::Dirichlet, right::Dirichlet)
    probvec_left = params(left)[1]
    probvec_right = params(right)[1]
    mvec = probvec_left .+ probvec_right
    mvec = mvec .- one(eltype(mvec))
    return Dirichlet(mvec)
end

BayesBase.probvec(dist::Dirichlet) = throw(ArgumentError("`probvec` is not defined for Dirichlet. Dirichlet represents a distribution over probability vectors, not a discrete distribution."))
BayesBase.std(dist::Dirichlet)     = map(sqrt, var(dist))

BayesBase.mean(::BroadcastFunction{typeof(log)}, dist::Dirichlet)      = digamma.(params(dist)[1]) .- digamma(sum(params(dist)[1]))
BayesBase.mean(::BroadcastFunction{typeof(clamplog)}, dist::Dirichlet) = digamma.((clamp(p, tiny, typemax(p)) for p in params(dist)[1])) .- digamma(sum(params(dist)[1]))

function BayesBase.compute_logscale(new_dist::Dirichlet, left_dist::Dirichlet, right_dist::Dirichlet)
    return logmvbeta(params(new_dist)[1]) - logmvbeta(params(left_dist)[1]) - logmvbeta(params(right_dist)[1])
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{Dirichlet}, x)
    l = length(getnaturalparameters(ef))
    return l == length(x) && !any(x -> x < zero(x), x) && sum(x) ≈ 1
end
# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Dirichlet}, η, conditioner) =
    isnothing(conditioner) && length(η) > 1 && all(isless.(-1, η)) && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{Dirichlet}, θ, conditioner) = isnothing(conditioner) && length(θ) > 1 && all(>(0), θ) && all(!isinf, θ)

function (::MeanToNatural{Dirichlet})(tuple_of_θ::Tuple{Any})
    (α,) = tuple_of_θ
    return (α - Ones{Float64}(length(α)),)
end

function (::NaturalToMean{Dirichlet})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    return (η + Ones{Float64}(length(η)),)
end

unpack_parameters(::Type{Dirichlet}, packed) = (packed,)

isbasemeasureconstant(::Type{Dirichlet}) = ConstantBaseMeasure()

getbasemeasure(::Type{Dirichlet}) = (x) -> one(Float64)
getsufficientstatistics(::Type{Dirichlet}) = (x -> map(log, x),)

getlogpartition(::NaturalParametersSpace, ::Type{Dirichlet}) = (η) -> begin
    (η1,) = unpack_parameters(Dirichlet, η)
    firstterm = mapreduce(x -> loggamma(x + 1), +, η1)
    secondterm = loggamma(sum(η1) + length(η1))
    return firstterm - secondterm
end

getfisherinformation(::NaturalParametersSpace, ::Type{Dirichlet}) =
    (η) -> begin
        (η1,) = unpack_parameters(Dirichlet, η)
        n = length(η1)
        return Diagonal(map(d -> trigamma(d + 1), η1)) - Ones{Float64}(n, n) * trigamma(sum(η1) + n)
    end

getgradlogpartition(::NaturalParametersSpace, ::Type{Dirichlet}) = (η) -> begin
    (η1,) = unpack_parameters(Dirichlet, η)
    return digamma.(η1 .+ 1) .- digamma(sum(η1 .+ 1))
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Dirichlet}) = (θ) -> begin
    (α,) = unpack_parameters(Dirichlet, θ)
    firstterm = mapreduce(x -> loggamma(x), +, α)
    secondterm = loggamma(sum(α))
    return firstterm - secondterm
end

getfisherinformation(::MeanParametersSpace, ::Type{Dirichlet}) = (θ) -> begin
    (α,) = unpack_parameters(Dirichlet, θ)
    n = length(α)
    return Diagonal(map(d -> trigamma(d), α)) - Ones{Float64}(n, n) * trigamma(sum(α))
end

getgradlogpartition(::MeanParametersSpace, ::Type{Dirichlet}) = (θ) -> begin
    (α,) = unpack_parameters(Dirichlet, θ)
    return digamma.(α) .- digamma(sum(α))
end
