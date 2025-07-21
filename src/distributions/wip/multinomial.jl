export Multinomial

import Distributions: Multinomial, probs
import StableRNGs: StableRNG
using StaticArrays
using LogExpFunctions

vague(::Type{<:Multinomial}, n::Int, dims::Int) = Multinomial(n, ones(dims) ./ dims)

probvec(dist::Multinomial) = probs(dist)

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

function pack_naturalparameters(dist::Multinomial)
    @inbounds p = params(dist)[2]
    return vmap(log, p / p[end])
end

unpack_naturalparameters(ef::ExponentialFamilyDistribution{Multinomial}) = (getnaturalparameters(ef),)

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Multinomial)
    n, _ = params(dist)
    return ExponentialFamilyDistribution(Multinomial, pack_naturalparameters(dist), n)
end

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Multinomial})
    expη = vmap(exp, getnaturalparameters(exponentialfamily))
    p = expη / sum(expη)
    return Multinomial(getconditioner(exponentialfamily), p)
end

check_valid_natural(::Type{<:Multinomial}, params) = length(params) >= 1

function check_valid_conditioner(::Type{<:Multinomial}, conditioner)
    isinteger(conditioner) && conditioner > 0
end

function isproper(exponentialfamily::ExponentialFamilyDistribution{Multinomial})
    logp = getnaturalparameters(exponentialfamily)
    n = getconditioner(exponentialfamily)
    return (n >= 1) && (length(logp) >= 1)
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{Multinomial})
    η = getnaturalparameters(exponentialfamily)
    n = getconditioner(exponentialfamily)
    return n * logsumexp(η)
end

function computeLogpartition(K, n)
    d = Multinomial(n, ones(K) ./ K)
    samples = unique(rand(StableRNG(1), d, 4000), dims = 2)
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

function fisherinformation(expfamily::ExponentialFamilyDistribution{Multinomial})
    η = getnaturalparameters(expfamily)
    n = getconditioner(expfamily)
    I = Matrix{Float64}(undef, length(η), length(η))
    seη = sum(vmap(exp, η))
    @inbounds for i in 1:length(η)
        I[i, i] = exp(η[i]) * (seη - exp(η[i])) / (seη)^2
        @inbounds for j in 1:i-1
            I[i, j] = -exp(η[i]) * exp(η[j]) / (seη)^2
            I[j, i] = I[i, j]
        end
    end
    return n * I
end

function fisherinformation(dist::Multinomial)
    n, p = params(dist)
    I = Matrix{Float64}(undef, length(p), length(p))
    @inbounds for i in 1:length(p)
        I[i, i] = (1 - p[i]) / p[i]
        @inbounds for j in 1:i-1
            I[i, j] = -1
            I[j, i] = I[i, j]
        end
    end
    return n * I
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{Multinomial, P, C, Safe}, x) where {P, C}
    n = Int(sum(x))
    return n == getconditioner(ef)
end

basemeasureconstant(::ExponentialFamilyDistribution{Multinomial}) = NonConstantBaseMeasure()
basemeasureconstant(::Type{<:Multinomial}) = NonConstantBaseMeasure()
basemeasure(ef::ExponentialFamilyDistribution{Multinomial}) = (x) -> basemeasure(ef, x)
function basemeasure(::ExponentialFamilyDistribution{Multinomial}, x::Vector)
    n = Int(sum(x))
    return factorial(n) / prod(@.factorial(x))
end

function basemeasure(::Multinomial, x::Vector)
    n = Int(sum(x))
    return factorial(n) / prod(@.factorial(x))
end

sufficientstatistics(::Union{<:ExponentialFamilyDistribution{Multinomial}, <:Multinomial}, x::Vector) = x
sufficientstatistics(ef::Union{<:ExponentialFamilyDistribution{Multinomial}, <:Multinomial}) =
    x -> sufficientstatistics(ef, x)
