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

function Base.prod(::ClosedProd, left::Multinomial, right::Multinomial)
    @assert left.n == right.n "$(left) and $(right) must have the same number of trials"
    trials = ntrials(left)
    K = length(left.p)
    η_left = getnaturalparameters(convert(KnownExponentialFamilyDistribution, left))
    η_right = getnaturalparameters(convert(KnownExponentialFamilyDistribution, right))

    naturalparameters = η_left + η_right
    sufficientstatistics = (x) -> x
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
