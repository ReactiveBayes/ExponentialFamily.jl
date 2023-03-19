export Multinomial

import Distributions: Multinomial, probs

vague(::Type{<:Multinomial}, n::Int, dims::Int) = Multinomial(n, ones(dims) ./ dims)

probvec(dist::Multinomial) = probs(dist) #return probability vector

function convert_eltype(::Type{Multinomial}, ::Type{T}, distribution::Multinomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Multinomial(n, convert(AbstractVector{T}, p))
end

prod_analytical_rule(::Type{<:Multinomial}, ::Type{<:Multinomial}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Multinomial, right::Multinomial)
    @assert left.n == right.n "$(left) and $(right) must have the same number of trials"

    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)

    return Multinomial(left.n, mvec ./ norm)
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Multinomial)
    n, p = params(dist)
    logprobabilities = log.(p)

    return ExponentialFamilyDistribution(Multinomial, logprobabilities, n)
end

function Base.convert(::Type{Distribution}, η::ExponentialFamilyDistribution{Multinomial})
    return Multinomial(getconditioner(η), softmax(getnaturalparameters(η)))
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

lognormalizer(::ExponentialFamilyDistribution{Multinomial}) = 0.0

function basemeasure(::Union{<:ExponentialFamilyDistribution{Multinomial}, <:Multinomial}, x)
    """
    x is a vector satisfying ∑x = n
    """
    n = Int(sum(x))
    return factorial(n) / prod(factorial.(x))
end

function plus(np1::ExponentialFamilyDistribution{Multinomial}, np2::ExponentialFamilyDistribution{Multinomial})
    if getconditioner(np1) == getconditioner(np2) && (first(size(getnaturalparameters(np1))) == first(size(getnaturalparameters(np2))))
        return Plus()
    else
        return Concat()
    end
end
