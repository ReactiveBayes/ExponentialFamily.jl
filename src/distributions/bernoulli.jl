export Bernoulli

import Distributions: Bernoulli, succprob, failprob, logpdf
import StatsFuns: logistic, logit

BayesBase.vague(::Type{<:Bernoulli}) = Bernoulli(0.5)
BayesBase.probvec(dist::Bernoulli) = (failprob(dist), succprob(dist))

BayesBase.default_prod_rule(::Type{<:Bernoulli}, ::Type{<:Bernoulli}) = PreserveTypeProd(Distribution)

function Base.prod(::PreserveTypeProd{Distribution}, left::Bernoulli, right::Bernoulli)
    left_p  = succprob(left)
    right_p = succprob(right)

    pprod = left_p * right_p
    norm  = pprod + (one(left_p) - left_p) * (one(right_p) - right_p)
    @assert norm > zero(norm) "Product of $(left) and $(right) results in non-normalizable distribution"
    return Bernoulli(pprod / norm)
end

BayesBase.default_prod_rule(::Type{<:Bernoulli}, ::Type{<:Categorical}) = PreserveTypeProd(Distribution)

function Base.prod(::PreserveTypeProd{Distribution}, left::Bernoulli, right::Categorical)
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

function BayesBase.compute_logscale(new_dist::Bernoulli, left_dist::Bernoulli, right_dist::Bernoulli)
    left_p = succprob(left_dist)
    right_p = succprob(right_dist)
    a = left_p * right_p + (one(left_p) - left_p) * (one(right_p) - right_p)
    return log(a)
end

function BayesBase.compute_logscale(new_dist::Categorical, left_dist::Bernoulli, right_dist::Categorical)
    p_left = probvec(left_dist)
    p_right = probvec(right_dist)

    Z = if length(p_left) >= length(p_right)
        dot(p_left, vcat(p_right..., zeros(length(p_left) - length(p_right))))
    else
        dot(p_right, vcat(p_left..., zeros(length(p_right) - length(p_left))))
    end

    return log(Z)
end

BayesBase.compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Bernoulli) =
    compute_logscale(new_dist, right_dist, left_dist)

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Bernoulli}, η, conditioner) = isnothing(conditioner) && (length(η) === 1) && (!isinf(first(η)))
isproper(::MeanParametersSpace, ::Type{Bernoulli}, θ, conditioner) = isnothing(conditioner) && (length(θ) === 1) && (0 <= first(θ) <= 1)

function (::MeanToNatural{Bernoulli})(tuple_of_θ::Tuple{Any})
    (p,) = tuple_of_θ
    return (logit(p),)
end

function (::NaturalToMean{Bernoulli})(tuple_of_η::Tuple{Any})
    (η₁,) = tuple_of_η
    return (logistic(η₁),)
end

function unpack_parameters(::Type{Bernoulli}, packed)
    return (first(packed),)
end

isbasemeasureconstant(::Type{Bernoulli}) = ConstantBaseMeasure()

getbasemeasure(::Type{Bernoulli}) = (x) -> oneunit(x)
getsufficientstatistics(::Type{Bernoulli}) = (identity,)

getlogpartition(::NaturalParametersSpace, ::Type{Bernoulli}) = (η) -> begin
    (η₁,) = unpack_parameters(Bernoulli, η)
    return -log(logistic(-η₁))
end

getfisherinformation(::NaturalParametersSpace, ::Type{Bernoulli}) = (η) -> begin
    (η₁,) = unpack_parameters(Bernoulli, η)
    f = logistic(-η₁)
    return SA[f * (one(f) - f);;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Bernoulli}) = (θ) -> begin
    (p,) = unpack_parameters(Bernoulli, θ)
    return -log(one(p) - p)
end

getfisherinformation(::MeanParametersSpace, ::Type{Bernoulli}) = (θ) -> begin
    (p,) = unpack_parameters(Bernoulli, θ)
    return SA[inv(p * (one(p) - p));;]
end
