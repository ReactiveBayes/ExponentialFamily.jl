export Erlang

import SpecialFunctions: logfactorial, digamma
import Distributions: Erlang, shape, scale, cov

Distributions.cov(dist::Erlang) = var(dist)

function mean(::typeof(log), dist::Erlang)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

vague(::Type{<:Erlang}) = Erlang(1, huge)

# Extensions of prod methods
prod_analytical_rule(::Type{<:Erlang}, ::Type{<:Erlang}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Erlang, right::Erlang)
    return Erlang(shape(left) + shape(right) - 1, (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

# Friendly function
function logpdf_sample_friendly(dist::Erlang)
    k, λ = params(dist)
    friendly = Erlang(k, λ)
    return (friendly, friendly)
end

# Natural parameters for the Erlang distribution
function check_valid_natural(::Type{<:Erlang}, params) 
    if (length(params) == 2) 
        true
    else
        false
    end
end
Base.convert(::Type{NaturalParameters}, dist::Erlang) = NaturalParameters(Erlang, [(shape(dist) - 1),-rate(dist)])

function Base.convert(::Type{Distribution}, params::NaturalParameters{Erlang})
    η = get_params(params)
    a = first(η)
    b = getindex(η, 2)
    return Erlang(Int64(a + 1), -1 / b)
end


function lognormalizer(params::NaturalParameters{Erlang})
    η = get_params(params)
    a = first(η)
    b = getindex(η, 2)
    return logfactorial(a) - (a + 1) * log(-b)
end

function isproper(params::NaturalParameters{Erlang})
    η = get_params(params)
    a = first(η)
    b = getindex(η,2)
    return (a >= tiny - 1) && (-b >= tiny)
end
