export Multinomial

import Distributions: Multinomial, probs
import StableRNGs: StableRNG

vague(::Type{<:Multinomial}, n::Int, dims::Int) = Multinomial(n, ones(dims) ./ dims)

probvec(dist::Multinomial) = probs(dist)

function convert_eltype(::Type{Multinomial}, ::Type{T}, distribution::Multinomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Multinomial(n, convert(AbstractVector{T}, p))
end

prod_closed_rule(::Type{<:Multinomial}, ::Type{<:Multinomial}) = ClosedProd()

function Base.prod(::ClosedProd, left::KnownExponentialFamilyDistribution{T}, right::KnownExponentialFamilyDistribution{T}) where {T <: Multinomial}
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
    basemeasure = (x) -> factorial(conditioner_left)^2 / (prod(factorial.(x)))^2
    logpartition = computeLogpartition(K, conditioner_left)
    supp = 0:conditioner_left
    return ExponentialFamilyDistribution(
        Float64,
        basemeasure,
        sufficientstatistics,
        naturalparameters,
        logpartition,
        supp
    )
end

function Base.prod(::ClosedProd, left::Multinomial, right::Multinomial)
    @assert left.n == right.n "$(left) and $(right) must have the same number of trials"
    trials = ntrials(left)
    K = length(left.p)
    η_left = getnaturalparameters(convert(KnownExponentialFamilyDistribution, left))
    η_right = getnaturalparameters(convert(KnownExponentialFamilyDistribution, right))

    naturalparameters = η_left + η_right
    sufficientstatistics = (x) -> x
    ## If number of trials is larger than 12 factorial will be problematic. Casting to BigInt will resolve the issue.
    ##TODO: fix this issue in future PRs
    basemeasure = (x) -> factorial(trials)^2 / (prod(factorial.(x)))^2
    logpartition = computeLogpartition(K, trials)
    supp = 0:trials
    return ExponentialFamilyDistribution(
        Float64,
        basemeasure,
        sufficientstatistics,
        naturalparameters,
        logpartition,
        supp
    )
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Multinomial)
    n, p = params(dist)
    η = log.(p / p[end])
    return KnownExponentialFamilyDistribution(Multinomial, η, n)
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Multinomial})
    expη = exp.(getnaturalparameters(exponentialfamily))
    p = expη / sum(expη)
    return Multinomial(getconditioner(exponentialfamily), p)
end

check_valid_natural(::Type{<:Multinomial}, params) = length(params) >= 1

function check_valid_conditioner(::Type{<:Multinomial}, conditioner)
    isinteger(conditioner) && conditioner > 0
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{Multinomial})
    logp = getnaturalparameters(exponentialfamily)
    n = getconditioner(exponentialfamily)
    return (n >= 1) && (length(logp) >= 1)
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Multinomial})
    η = getnaturalparameters(exponentialfamily)
    n = getconditioner(exponentialfamily)
    return n * log(sum(exp.(η)))
end

function basemeasure(::Union{<:KnownExponentialFamilyDistribution{Multinomial}, <:Multinomial}, x)
    n = Int(sum(x))
    return factorial(n) / prod(factorial.(x))
end

function computeLogpartition(K, n)
    d = Multinomial(n, ones(K) ./ K)
    samples = unique(rand(StableRNG(1), d, 4000), dims = 2)
    samples = [samples[:, i] for i in 1:size(samples, 2)]
    return let samples = samples
        (η) -> begin
            result = mapreduce(+, samples) do xi
                return (factorial(n) / prod(factorial.(xi)))^2 * exp(η' * xi)
            end
            return log(result)
        end
    end
end

function fisherinformation(expfamily::KnownExponentialFamilyDistribution{Multinomial})
    η = getnaturalparameters(expfamily)
    n = getconditioner(expfamily)
    I = Matrix{Float64}(undef, length(η), length(η))
    @inbounds for i in 1:length(η)
        I[i, i] = exp(η[i]) * (sum(exp.(η)) - exp(η[i])) / (sum(exp.(η)))^2
        for j in 1:i-1
            I[i, j] = -exp(η[i]) * exp(η[j]) / (sum(exp.(η)))^2
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
        for j in 1:i-1
            I[i, j] = -1
            I[j, i] = I[i, j]
        end
    end
    return n * I
end
