export Geometric

import Distributions: Geometric, succprob, failprob
using DomainSets
using StaticArrays

## We use the variant of Geometric distribution that models k failures before the first success
BayesBase.vague(::Type{<:Geometric}) = Geometric(Float64(tiny))
BayesBase.probvec(dist::Geometric) = (failprob(dist), succprob(dist))

BayesBase.default_prod_rule(::Type{<:Geometric}, ::Type{<:Geometric}) = PreserveTypeProd(Distribution)

BayesBase.prod(::PreserveTypeProd{Distribution}, left::Geometric, right::Geometric) =
    Geometric(succprob(left) + succprob(right) - succprob(left) * succprob(right))

# Natural parametrization

getsupport(::Type{Geometric}) = DomainSets.NaturalNumbers()

isproper(::NaturalParametersSpace, ::Type{Geometric}, η, conditioner) =
    isnothing(conditioner) && length(η) === 1 && all(!isinf, η) && all(!isnan, η) && all(<=(0), η)
isproper(::MeanParametersSpace, ::Type{Geometric}, θ, conditioner) =
    isnothing(conditioner) && length(θ) === 1 && all(>(0), θ) && all(<=(1), θ) && all(!isinf, θ) && all(!isnan, θ)

function (::MeanToNatural{Geometric})(tuple_of_θ::Tuple{Any})
    (p,) = tuple_of_θ
    return (log(one(p) - p),)
end

function (::NaturalToMean{Geometric})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    return (one(η) - exp(η),)
end

unpack_parameters(::Type{Geometric}, packed) = (first(packed),)

isbasemeasureconstant(::Type{Geometric}) = ConstantBaseMeasure()

getbasemeasure(::Type{Geometric}) = (x) -> one(x)
getsufficientstatistics(::Type{Geometric}) = (identity,)

getlogpartition(::NaturalParametersSpace, ::Type{Geometric}) = (η) -> begin
    (η1,) = unpack_parameters(Geometric, η)
    return -log(one(η1) - exp(η1))
end

getfisherinformation(::NaturalParametersSpace, ::Type{Geometric}) = (η) -> begin
    (η1,) = unpack_parameters(Geometric, η)
    return SA[exp(η1) / (one(η1) - exp(η1))^2;;]
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Geometric}) = (η) -> begin
    (η1,) = unpack_parameters(Geometric, η)
    return SA[exp(η1) / (one(η1) - exp(η1));]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Geometric}) = (θ) -> begin
    (p,) = unpack_parameters(Geometric, θ)
    return -log(p)
end

getfisherinformation(::MeanParametersSpace, ::Type{Geometric}) = (θ) -> begin
    (p,) = unpack_parameters(Geometric, θ)
    return SA[one(p) / (p^2 * (one(p) - p));;]
end

getgradlogpartition(::MeanParametersSpace, ::Type{Geometric}) = (θ) -> begin
    (p,) = unpack_parameters(Geometric, θ)
    return SA[one(p) / (p^2 - p);]
end
