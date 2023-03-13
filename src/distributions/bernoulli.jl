export Bernoulli

import Distributions: Bernoulli, Distribution, succprob, failprob, logpdf
import StatsFuns: logistic

vague(::Type{<:Bernoulli}) = Bernoulli(0.5)

probvec(dist::Bernoulli) = (failprob(dist), succprob(dist))

prod_analytical_rule(::Type{<:Bernoulli}, ::Type{<:Bernoulli}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Bernoulli, right::Bernoulli)
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end

prod_analytical_rule(::Type{<:Bernoulli}, ::Type{<:Categorical}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Bernoulli, right::Categorical)

    # get probability vectors
    p_left = probvec(left)
    p_right = probvec(right)

    # find the maximum length of both arguments
    max_length = max(length(p_left), length(p_right))

    # preallocate the result
    p_new = zeros(promote_samplefloattype(p_left, p_right), max_length)

    # Create extended versions of the left and the right prob vectors
    # Do so by appending `0` with the Iterators.repeated, branch and allocation free
    e_left  = Iterators.flatten((p_left, Iterators.repeated(0, max(0, length(p_right) - length(p_left)))))
    e_right = Iterators.flatten((p_right, Iterators.repeated(0, max(0, length(p_left) - length(p_right)))))

    for (i, l, r) in zip(eachindex(p_new), e_left, e_right)
        @inbounds p_new[i] = l * r
    end

    # return categorical with normalized probability vector
    return Categorical(normalize!(p_new, 1))
end

function compute_logscale(new_dist::Bernoulli, left_dist::Bernoulli, right_dist::Bernoulli)
    left_p = succprob(left_dist)
    right_p = succprob(right_dist)
    a = left_p * right_p + (one(left_p) - left_p) * (one(right_p) - right_p)
    return log(a)
end

function compute_logscale(new_dist::Categorical, left_dist::Bernoulli, right_dist::Categorical)
    # get probability vectors
    p_left = probvec(left_dist)
    p_right = probvec(right_dist)

    # find length of new vector and compute entries
    Z = if length(p_left) >= length(p_right)
        dot(p_left, vcat(p_right..., zeros(length(p_left) - length(p_right))))
    else
        dot(p_right, vcat(p_left..., zeros(length(p_right) - length(p_left))))
    end

    # return log normalization constant
    return log(Z)
end

compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Bernoulli) =
    compute_logscale(new_dist, right_dist, left_dist)

function lognormalizer(params::NaturalParameters{Bernoulli})
    return -log(logistic(-first(get_params(params))))
end

function Base.convert(::Type{Distribution}, params::NaturalParameters{Bernoulli})
    logprobability = getindex(get_params(params), 1)
    return Bernoulli(exp(logprobability) / (1 + exp(logprobability)))
end

function Base.convert(::Type{NaturalParameters}, dist::Bernoulli)
    @assert !(succprob(dist) ≈ 1) "Bernoulli natural parameters are not defiend for p = 1."
    NaturalParameters(Bernoulli, [log(succprob(dist) / (1 - succprob(dist)))])
end

isproper(params::NaturalParameters{Bernoulli}) = true

check_valid_natural(::Type{<:Bernoulli}, params) = (length(params) === 1)

basemeasure(T::Union{<:NaturalParameters{Bernoulli}, <:Bernoulli}, x) = 1.0
