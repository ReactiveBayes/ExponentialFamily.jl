export Rayleigh

import Distributions: Rayleigh, params

vague(::Type{<:Rayleigh}) = Rayleigh(1e12)

# prod_analytical_rule(::Type{<:Rayleigh}, ::Type{<:Rayleigh}) = ProdAnalyticalRuleAvailable()

# function Base.prod(::ProdAnalytical, left::Rayleigh, right::Rayleigh)
#     left_a, left_b   = params(left)
#     right_a, right_b = params(right)
#     T                = promote_samplefloattype(left, right)
#     return Rayleigh(left_a + right_a - one(T), left_b + right_b - one(T))
# end

function isproper(params::NaturalParameters{Rayleigh})
    η = first(get_params(params))
    return (η < 0)
end

function Base.convert(::Type{NaturalParameters}, dist::Rayleigh)
    σ = first(params(dist))
    NaturalParameters(Rayleigh, [-1 / (2σ^2)])
end

function Base.convert(::Type{Distribution}, params::NaturalParameters{Rayleigh})
    η = first(get_params(params))
    return Rayleigh(sqrt(-1 / (2η)))
end

check_valid_natural(::Type{<:Rayleigh}, v) = length(v) === 1

lognormalizer(params::NaturalParameters{Rayleigh}) = log(-2first(get_params(params)))

basemeasure(::Union{<:NaturalParameters{Rayleigh}, <:Rayleigh}, x) = x

plus(::NaturalParameters{Rayleigh}, ::NaturalParameters{Rayleigh}) = Plus()
