export Chisq

import SpecialFunctions: loggamma
import Distributions: Chisq, params, dof
using StaticArrays
using DomainSets

default_prod_rule(::Type{<:Chisq}, ::Type{<:Chisq}) = PreserveTypeProd(ExponentialFamilyDistribution)

function Base.prod(::PreserveTypeProd{ExponentialFamilyDistribution}, left::Chisq, right::Chisq)
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)

    return prod(PreserveTypeProd(ExponentialFamilyDistribution),ef_left, ef_right)
end

# # NOTE: The product of two Chisq distributions is NOT a Chisq distribution.
function Base.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Chisq}
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right
    basemeasure = (x) -> exp(-x)
    sufficientstatistics = (x) -> SA[log(x)]
    logpartition = (η) -> loggamma(η[1] + 1)
    supp = Distributions.support(T)

    attributes = ExponentialFamilyDistributionAttributes(basemeasure,sufficientstatistics,logpartition,supp)

    return ExponentialFamilyDistribution(
        Univariate,
        naturalparameters,
        nothing,
        attributes
    )
end

function compute_logscale(new_dist::Chisq, left_dist::Chisq, right_dist::Chisq)
    lp = getlogpartition(MeanParametersSpace(), Chisq)
    return lp(params(new_dist)...) - lp(params(left_dist)...) - lp(params(right_dist)...)
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Chisq}, η, conditioner) = isnothing(conditioner) && length(η) === 1 && all(>(-1/2), η) && all(!isinf, η)
isproper(::MeanParametersSpace, ::Type{Chisq}, θ, conditioner) = isnothing(conditioner) && length(θ) === 1 && all(>(0), θ) && all(isinteger, θ) && all(!isinf, θ)

function (::MeanToNatural{Chisq})(tuple_of_θ::Tuple{Any})
    (ν, ) = tuple_of_θ
    return ((ν/2) - one(ν), )
end

function (::NaturalToMean{Chisq})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    return (2*(η + one(η)), )
end

unpack_parameters(::Type{Chisq}, packed) = (first(packed), )

isbasemeasureconstant(::Type{Chisq}) = NonConstantBaseMeasure()

getbasemeasure(::Type{Chisq}) = (x) -> exp(-x/2)
getsufficientstatistics(::Type{Chisq}) = (log, )

getlogpartition(::NaturalParametersSpace, ::Type{Chisq}) = (η) -> begin
    (η1, ) = unpack_parameters(Chisq, η)
    o = one(η1)
    return loggamma(η1 + o) + (η1 + o) * logtwo
end

getfisherinformation(::NaturalParametersSpace, ::Type{Chisq}) = (η) -> begin
    (η1, )= unpack_parameters(Chisq, η)
    SA[trigamma(η1 + one(η1));;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Chisq}) = (θ) -> begin
    (ν, ) = unpack_parameters(Chisq, θ)
    return loggamma(ν/2) + (ν/2)*logtwo
end

getfisherinformation(::MeanParametersSpace, ::Type{Chisq}) = (θ) -> begin
    (ν,  ) = unpack_parameters(Chisq, θ)
    return SA[trigamma(ν /2) * QUARTER;;]
end