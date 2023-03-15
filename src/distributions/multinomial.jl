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
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)

    @assert left.n == right.n "$(left) and $(right) must have the same number of trials"
    return Multinomial(left.n, mvec ./ norm)
end

# Standard parameters to natural parameters
function Base.convert(::Type{NaturalParameters}, dist::Multinomial)
    n, p = params(dist)
    logprobabilities = log.(p)

    # The params of NaturalParameters(Multinomial) is a Tuple (n, log(probvec))
    return NaturalParameters(Multinomial, logprobabilities, n)
end

function Base.convert(::Type{Distribution}, η::NaturalParameters{Multinomial})
    @assert !isnothing(get_conditioner(η)) "Number of trials should be supplied as a conditioner to convert back. Multinomial natural parameters should always be created with conditioiner"
    return Multinomial(get_conditioner(η), softmax(get_params(η)))
end
check_valid_natural(::Type{<:Multinomial}, params) = length(params) >= 1

function isproper(params::NaturalParameters{Multinomial})
    @assert !isnothing(get_conditioner(params)) "Conditioner should be supplied"
    logp = get_params(params)
    n = get_conditioner(params)
    return (n >= 1) && (length(logp) >= 2)
end

lognormalizer(::NaturalParameters{Multinomial}) = 0.0

function basemeasure(::Union{<:NaturalParameters{Multinomial}, <:Multinomial}, x)
    """
    x is a vector satisfying ∑x = n
    """
    n = Int(sum(x))
    return factorial(n) / prod(factorial.(x))
end

function plus(np1::NaturalParameters{Multinomial}, np2::NaturalParameters{Multinomial})
    if get_conditioner(np1) == get_conditioner(np2)
        return Plus()
    else
        return Concat()
    end
end

# function Base.:+(left::NaturalParameters{Multinomial}, right::NaturalParameters{Multinomial})
#     η_left = get_params(left)
#     η_right = get_params(right)

#     @assert first(η_left) == first(η_right) "$(left) and $(right) must have the same number of trials"
#     return NaturalParameters(Multinomial, (first(η_left), last(η_left) + last(η_right)))
# end

# function Base.:-(left::NaturalParameters{Multinomial}, right::NaturalParameters{Multinomial})
#     η_left = get_params(left)
#     η_right = get_params(right)

#     @assert first(η_left) == first(η_right) "$(left) and $(right) must have the same number of trials"
#     return NaturalParameters(Multinomial, (first(η_left), last(η_left) - last(η_right)))
# end
