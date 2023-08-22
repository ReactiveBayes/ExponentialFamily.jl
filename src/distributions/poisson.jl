export Poisson

import SpecialFunctions: besseli
import Distributions: Poisson, shape, scale, cov
using DomainSets
using StaticArrays

Distributions.cov(dist::Poisson) = var(dist)

default_prod_rule(::Type{<:Poisson}, ::Type{<:Poisson}) = PreserveTypeProd(ExponentialFamilyDistribution)
function support(::Type{Poisson})
    return DomainSets.NaturalNumbers()
end
# NOTE: The product of two Poisson distributions is NOT a Poisson distribution.
function Base.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Poisson}
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right
    basemeasure = (x) -> one(x) / gamma(x+one(x))^2
    sufficientstatistics = (identity, )
    logpartition = (η) -> log(abs(besseli(0, 2 * exp(first(η) / 2))))
    supp = DomainSets.NaturalNumbers()
    attributes = ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics,logpartition,supp)

    return ExponentialFamilyDistribution(
        Univariate,
        naturalparameters,
        nothing,
        attributes
    )
end

function Base.prod(::PreserveTypeProd{ExponentialFamilyDistribution}, left::Poisson, right::Poisson)
    ef_left = convert(ExponentialFamilyDistribution, left)
    ef_right = convert(ExponentialFamilyDistribution, right)

    return prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_left, ef_right)
end

function logpdf_sample_friendly(dist::Poisson)
    λ = params(dist)
    friendly = Poisson(λ)
    return (friendly, friendly)
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Poisson}, η, conditioner) = isnothing(conditioner) && length(η) === 1 && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{Poisson}, θ, conditioner) = isnothing(conditioner) && length(θ) === 1 && all(>(0), θ)  && all(!isinf, θ)&& all(!isnan, θ)

function (::MeanToNatural{Poisson})(tuple_of_θ::Tuple{Any})
    (λ, ) = tuple_of_θ
    return (log(λ), )
end

function (::NaturalToMean{Poisson})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    return (exp(η), )
end

unpack_parameters(::Type{Poisson}, packed) = (first(packed), )

isbasemeasureconstant(::Type{Poisson}) = NonConstantBaseMeasure()

getbasemeasure(::Type{Poisson}) = (x) -> one(x) / factorial(x)
getsufficientstatistics(::Type{Poisson}) = (identity, )

getlogpartition(::NaturalParametersSpace, ::Type{Poisson}) = (η) -> begin
    (η1, ) = unpack_parameters(Poisson, η)
    return exp(η1)
end

getfisherinformation(::NaturalParametersSpace, ::Type{Poisson}) = (η) -> begin
    (η1, ) = unpack_parameters(Poisson, η)
    SA[exp(η1);;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Poisson}) = (θ) -> begin
    (λ, ) = unpack_parameters(Poisson, θ)
    return λ
end

getfisherinformation(::MeanParametersSpace, ::Type{Poisson}) = (θ) -> begin
    (λ,  ) = unpack_parameters(Poisson, θ)
    return SA[1/λ;;]
end


# check_valid_natural(::Type{<:Poisson}, params) = isequal(length(params), 1)

# pack_naturalparameters(dist::Poisson) = [log(rate(dist))]
# function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Poisson})
#     η = getnaturalparameters(ef)
#     @inbounds η1 = η[1]
#     return (η1,)
# end

# Base.convert(::Type{ExponentialFamilyDistribution}, dist::Poisson) =
#     ExponentialFamilyDistribution(Poisson, pack_naturalparameters(dist))

# function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Poisson})
#     (η,) = unpack_naturalparameters(exponentialfamily)
#     return Poisson(exp(η))
# end

# logpartition(exponentialfamily::ExponentialFamilyDistribution{Poisson}) =
#     exp(first(unpack_naturalparameters(exponentialfamily)))

# function isproper(exponentialfamily::ExponentialFamilyDistribution{Poisson})
#     (η,) = unpack_naturalparameters(exponentialfamily)
#     η isa Number && !isnan(η) && !isinf(η)
# end

# fisherinformation(exponentialfamily::ExponentialFamilyDistribution{Poisson}) =
#     SA[exp(first(unpack_naturalparameters(exponentialfamily)));;]

# fisherinformation(dist::Poisson) = SA[1 / rate(dist);;]

# function support(::ExponentialFamilyDistribution{Poisson})
#     return DomainSets.NaturalNumbers()
# end

# basemeasureconstant(::ExponentialFamilyDistribution{Poisson}) = NonConstantBaseMeasure()
# basemeasureconstant(::Type{<:Poisson}) = NonConstantBaseMeasure()

# basemeasure(ef::Union{<:ExponentialFamilyDistribution{Poisson}, <:Poisson}) = x -> basemeasure(ef, x)
# function basemeasure(::Union{<:ExponentialFamilyDistribution{Poisson}, <:Poisson}, x::Real)
#     return one(x) / factorial(x)
# end

# sufficientstatistics(ef::ExponentialFamilyDistribution) = (x) -> sufficientstatistics(ef, x)
# function sufficientstatistics(::Union{<:ExponentialFamilyDistribution{Poisson}, <:Poisson}, x::Real)
#     return SA[x]
# end
