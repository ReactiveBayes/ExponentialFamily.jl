export Multinomial

import Distributions: Multinomial, probs
using StaticArrays
using LogExpFunctions

BayesBase.vague(::Type{<:Multinomial}, n::Int, dims::Int) = Multinomial(n, ones(dims) ./ dims)

BayesBase.probvec(dist::Multinomial) = probs(dist)

function convert_eltype(::Type{Multinomial}, ::Type{T}, distribution::Multinomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Multinomial(n, convert(AbstractVector{T}, p))
end

BayesBase.default_prod_rule(::Type{<:Multinomial}, ::Type{<:Multinomial}) = PreserveTypeProd(ExponentialFamilyDistribution)

# NOTE: The product of two Multinomial distributions is NOT a Multinomial distribution.
function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Multinomial}
    conditioner_left = getconditioner(left)
    conditioner_right = getconditioner(right)
    @assert conditioner_left == conditioner_right "$(left) and $(right) must have the same conditioner"

    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    K = length(η_left)
    naturalparameters = η_left + η_right
    sufficientstatistics = (x) -> x
    ## If conditioner is larger than 12 factorial will be problematic. Casting to BigInt will resolve the issue.
    ##TODO: fix this issue in future PRs
    basemeasure = (x) -> factorial(conditioner_left)^2 / (prod(@.factorial(x)))^2
    logpartition = computeLogpartition(K, conditioner_left)
    supp = 0:conditioner_left
    return ExponentialFamilyDistribution(
        Multivariate,
        naturalparameters,
        nothing,
        basemeasure,
        sufficientstatistics,
        logpartition,
        supp
    )
end

function BayesBase.prod(::ClosedProd, left::T, right::T) where {T <: Multinomial}
    @assert left.n == right.n "$(left) and $(right) must have the same number of trials"
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)
    return prod(ClosedProd(), ef_left, ef_right)
end

check_valid_natural(::Type{<:Multinomial}, params) = length(params) >= 1

function check_valid_conditioner(::Type{<:Multinomial}, conditioner)
    isinteger(conditioner) && conditioner > 0
end

function isproper(::NaturalParametersSpace, ::Type{Multinomial}, natural_parameters::AbstractVector{<:Real}, conditioner::Int)
   return (conditioner >= 1) && (length(natural_parameters) >= 1)
end

function unpack_parameters(::Type{Multinomial}, packed::AbstractVector, conditioner)
    @show packed
    return (packed,)
end

function pack_parameters(::Type{Multinomial}, unpacked::Tuple{<:AbstractVector})
    return first(unpacked)
end

function separate_conditioner(::Type{Multinomial}, params)
    ndims, success_probs = params
    return ((success_probs,), ndims)
end

function join_conditioner(::Type{Multinomial}, cparams, conditioner)
    (succprob,) = cparams
    ntrials = conditioner
    return (ntrials, succprob)
end

function (::MeanToNatural{Multinomial})(parameters::Tuple{<:AbstractVector}, conditioner)
    (succprob,) = parameters
    return (log.(succprob),)
end

function (::NaturalToMean{Multinomial})(natural_parameters::Tuple{<:AbstractVector}, conditioner)
    (log_probs,) = natural_parameters
    return (exp.(log_probs),)
end

getsufficientstatistics(::Type{Multinomial}, _) = (identity,)
getgradlogpartition(::NaturalParametersSpace, ::Type{Multinomial}, conditioner::Int) = (η) -> zeros(length(η))

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Multinomial})
    η = getnaturalparameters(exponentialfamily)
    n = getconditioner(exponentialfamily)
    return n * logsumexp(η)
end

getfisherinformation(::NaturalParametersSpace, ::Type{Multinomial}, conditioner::Int) = (η) -> begin
    I = Matrix{Float64}(undef, length(η), length(η))
    seη = mapreduce(exp, +, η)
    @inbounds for i in 1:length(η)
        I[i, i] = exp(η[i]) * (seη - exp(η[i])) / (seη)^2
        @inbounds for j in 1:(i-1)
            I[i, j] = -exp(η[i]) * exp(η[j]) / (seη)^2
            I[j, i] = I[i, j]
        end
    end
    return conditioner * I
end

getfisherinformation(::MeanParametersSpace, ::Type{Multinomial}, conditioner::Int) = (θ) -> begin
    I = Matrix{Float64}(undef, length(θ), length(θ))
    @inbounds for i in 1:length(θ)
        I[i, i] = (1 - θ[i]) / θ[i]
        @inbounds for j in 1:(i-1)
            I[i, j] = -1
            I[j, i] = I[i, j]
        end
    end
    return conditioner * I
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{Multinomial, P, C, S}, x) where {P, C, S}
    n = Int(sum(x))
    return n == getconditioner(ef)
end

basemeasureconstant(::ExponentialFamilyDistribution{Multinomial}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:Multinomial}) = NonConstantBaseMeasure()
basemeasure(ef::ExponentialFamilyDistribution{Multinomial}) = (x) -> basemeasure(ef, x)

sufficientstatistics(::Union{<:ExponentialFamilyDistribution{Multinomial}, <:Multinomial}, x::Vector) = x
sufficientstatistics(ef::Union{<:ExponentialFamilyDistribution{Multinomial}, <:Multinomial}) =
    x -> sufficientstatistics(ef, x)

getbasemeasure(::Type{Multinomial}, ntrials) = (x) -> begin
    n = Int(sum(x))
    return factorial(n) / prod(@.factorial(x))
end
getlogbasemeasure(::Type{Multinomial}, ntrials) = (x) -> log(getbasemeasure(Multinomial, ntrials)(x)) # TODO change with loggamma
getlogpartition(::NaturalParametersSpace, ::Type{Multinomial}, conditioner::Int) = (η) -> zeros(length(η))