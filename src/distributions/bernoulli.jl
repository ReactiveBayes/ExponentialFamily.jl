export Bernoulli

import Distributions: Bernoulli, succprob, failprob, logpdf
import StatsFuns: logistic

vague(::Type{<:Bernoulli}) = Bernoulli(0.5)

probvec(dist::Bernoulli) = (failprob(dist), succprob(dist))

prod_analytical_rule(::Type{<:Bernoulli}, ::Type{<:Bernoulli}) = ClosedProd()
function Base.prod(::ClosedProd, left::Bernoulli, right::Bernoulli)
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > 0 "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end

prod_analytical_rule(::Type{<:Bernoulli}, ::Type{<:Categorical}) = ClosedProd()

function Base.prod(::ClosedProd, left::Bernoulli, right::Categorical)
    p_left = probvec(left)
    p_right = probvec(right)

    max_length = max(length(p_left), length(p_right))

    p_new = zeros(promote_samplefloattype(p_left, p_right), max_length)

    e_left  = Iterators.flatten((p_left, Iterators.repeated(0, max(0, length(p_right) - length(p_left)))))
    e_right = Iterators.flatten((p_right, Iterators.repeated(0, max(0, length(p_left) - length(p_right)))))

    for (i, l, r) in zip(eachindex(p_new), e_left, e_right)
        @inbounds p_new[i] = l * r
    end

    return Categorical(normalize!(p_new, 1))
end

function compute_logscale(new_dist::Bernoulli, left_dist::Bernoulli, right_dist::Bernoulli)
    left_p = succprob(left_dist)
    right_p = succprob(right_dist)
    a = left_p * right_p + (one(left_p) - left_p) * (one(right_p) - right_p)
    return log(a)
end

function compute_logscale(new_dist::Categorical, left_dist::Bernoulli, right_dist::Categorical)
    p_left = probvec(left_dist)
    p_right = probvec(right_dist)

    Z = if length(p_left) >= length(p_right)
        dot(p_left, vcat(p_right..., zeros(length(p_left) - length(p_right))))
    else
        dot(p_right, vcat(p_left..., zeros(length(p_right) - length(p_left))))
    end

    return log(Z)
end

compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Bernoulli) =
    compute_logscale(new_dist, right_dist, left_dist)

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Bernoulli})
    return -log(logistic(-first(getnaturalparameters(exponentialfamily))))
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Bernoulli})
    logprobability = getindex(getnaturalparameters(exponentialfamily), 1)
    return Bernoulli(exp(logprobability) / (1 + exp(logprobability)))
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Bernoulli)
    @assert !(succprob(dist) ≈ 1) "Bernoulli natural parameters are not defiend for p = 1."
    KnownExponentialFamilyDistribution(Bernoulli, [log(succprob(dist) / (1 - succprob(dist)))])
end

isproper(exponentialfamily::KnownExponentialFamilyDistribution{Bernoulli}) = true

check_valid_natural(::Type{<:Bernoulli}, params) = (length(params) === 1)

basemeasure(T::Union{<:KnownExponentialFamilyDistribution{Bernoulli}, <:Bernoulli}, x) = 1.0
