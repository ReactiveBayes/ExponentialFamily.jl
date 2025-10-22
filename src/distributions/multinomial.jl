export Multinomial

import Distributions: Multinomial, probs
using StaticArrays
using LogExpFunctions

BayesBase.vague(::Type{<:Multinomial}, n::Int, dims::Int) = Multinomial(n, ones(dims) ./ dims)

BayesBase.convert_paramfloattype(::Type{T}, distribution::Multinomial) where {T <: Real} =
    Multinomial(distribution.n, convert(AbstractVector{T}, probvec(distribution)))

BayesBase.default_prod_rule(::Type{<:Multinomial}, ::Type{<:Multinomial}) = PreserveTypeProd(ExponentialFamilyDistribution)

function __compute_logpartition_multinomial_product(K::Int, n::Int)
    d = vague(Multinomial, n, K)
    samples = unique(rand(d, 4000), dims = 2)
    samples = [samples[:, i] for i in 1:size(samples, 2)]
    return let samples = samples
        (η) -> begin
            result = mapreduce(+, samples) do xi
                return (factorial(n) / prod(@.factorial(xi)))^2 * exp(η' * xi)
            end
            return log(result)
        end
    end
end

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
    sufficientstatistics = (identity,)

    logbasemeasure = (x) -> 2 * loggamma(conditioner_left + 1) - 2 * sum(loggamma.(x .+ 1))
    basemeasure = (x) -> exp(logbasemeasure(x))

    # Create log partition function that takes natural parameters as input
    logpartition = __compute_logpartition_multinomial_product(K, conditioner_left)

    supp = 0:conditioner_left

    attributes = ExponentialFamilyDistributionAttributes(
        basemeasure,
        sufficientstatistics,
        logpartition,
        supp
    )

    return ExponentialFamilyDistribution(
        Multivariate,
        naturalparameters,
        η_left,
        attributes
    )
end

BayesBase.probvec(dist::Multinomial) = probs(dist)

check_valid_natural(::Type{<:Multinomial}, params) = length(params) >= 1

function check_valid_conditioner(::Type{<:Multinomial}, conditioner)
    isinteger(conditioner) && conditioner > 0
end

function isproper(::NaturalParametersSpace, ::Type{Multinomial}, natural_parameters::AbstractVector{<:Real}, conditioner::Int)
    return (conditioner >= 1) && (length(natural_parameters) >= 1)
end

function unpack_parameters(::Type{Multinomial}, packed, conditioner)
    return (packed,)
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
    pk = last(succprob)
    return (map(pi -> log(pi / pk), succprob),)
    # return (log.(succprob),)
end

function (::NaturalToMean{Multinomial})(natural_parameters::Tuple{<:AbstractVector}, conditioner)
    (log_probs,) = natural_parameters
    return (softmax(log_probs),)
end

getsufficientstatistics(::Type{Multinomial}, _) = (identity,)

isbasemeasureconstant(::Type{Multinomial}) = NonConstantBaseMeasure()
getbasemeasure(::Type{Multinomial}, ntrials) = (x) -> factorial(sum(x)) / prod(@.factorial(x))
getlogbasemeasure(::Type{Multinomial}, ntrials) = (x) -> loggamma(sum(x) + 1) - sum(loggamma.(x .+ 1))

getlogpartition(::NaturalParametersSpace, ::Type{Multinomial}, conditioner::Int) = (η) -> conditioner * logsumexp(η)

getgradlogpartition(::NaturalParametersSpace, ::Type{Multinomial}, conditioner::Int) = (η) -> begin
    sumη = mapreduce(exp, +, η)
    return map(d -> conditioner * exp(d) / sumη, η)
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
