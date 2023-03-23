export Multinomial

import Distributions: Multinomial, probs

vague(::Type{<:Multinomial}, n::Int, dims::Int) = Multinomial(n, ones(dims) ./ dims)

probvec(dist::Multinomial) = probs(dist) #return probability vector

function convert_eltype(::Type{Multinomial}, ::Type{T}, distribution::Multinomial{R}) where {T <: Real, R <: Real}
    n, p = params(distribution)
    return Multinomial(n, convert(AbstractVector{T}, p))
end

prod_closed_rule(::Type{<:Multinomial}, ::Type{<:Multinomial}) = ClosedProd()

function Base.prod(::ClosedProd, left::Multinomial, right::Multinomial)
    @assert left.n == right.n "$(left) and $(right) must have the same number of trials"

    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)

    return Multinomial(left.n, mvec ./ norm)
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Multinomial)
    n, p = params(dist)
    logprobabilities = log.(p)

    return KnownExponentialFamilyDistribution(Multinomial, logprobabilities, n)
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Multinomial})
    return Multinomial(getconditioner(exponentialfamily), softmax(getnaturalparameters(exponentialfamily)))
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

logpartition(::KnownExponentialFamilyDistribution{Multinomial}) = 0.0

function basemeasure(::Union{<:KnownExponentialFamilyDistribution{Multinomial}, <:Multinomial}, x)
    """
    x is a vector satisfying ∑x = n
    """
    n = Int(sum(x))
    return factorial(n) / prod(factorial.(x))
end


