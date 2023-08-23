export Rayleigh

import Distributions: Rayleigh, params
using DomainSets
using StaticArrays

vague(::Type{<:Rayleigh}) = Rayleigh(Float64(huge))

# NOTE: The product of two Rayleigh distributions is NOT a Rayleigh distribution.
function Base.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: Rayleigh}
    η1 = getnaturalparameters(left)
    η2 = getnaturalparameters(right)
    naturalparameters = η1 + η2
    basemeasure = (x) -> 4 * x^2 / sqrt(pi)
    sufficientstatistics = (x -> x^2, )
    logpartition = (η) -> (-3 / 2)log(-first(η))
    support = DomainSets.HalfLine()
    attributes = ExponentialFamilyDistributionAttributes(basemeasure,sufficientstatistics,logpartition,support)

    return ExponentialFamilyDistribution(
        Univariate,
        naturalparameters,
        nothing,
        attributes
    )
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{Rayleigh}, η, conditioner) = isnothing(conditioner) && length(η) === 1 && all( <(0), η) && all(!isinf, η)
isproper(::MeanParametersSpace, ::Type{Rayleigh}, θ, conditioner) = isnothing(conditioner) && length(θ) === 1 && all(>(0), θ)  && all(!isinf, θ)

function (::MeanToNatural{Rayleigh})(tuple_of_θ::Tuple{Any})
    (σ, ) = tuple_of_θ
    return (-1/(2*σ^2), )
end

function (::NaturalToMean{Rayleigh})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    return (sqrt(-1 / (2η)), )
end

unpack_parameters(::Type{Rayleigh}, packed) = (first(packed), )

isbasemeasureconstant(::Type{Rayleigh}) = NonConstantBaseMeasure()

getbasemeasure(::Type{Rayleigh}) = identity
getsufficientstatistics(::Type{Rayleigh}) = (x -> x^2, )

getlogpartition(::NaturalParametersSpace, ::Type{Rayleigh}) = (η) -> begin
    (η1, ) = unpack_parameters(Rayleigh, η)
    return -log(-2 * η1)
end

getfisherinformation(::NaturalParametersSpace, ::Type{Rayleigh}) = (η) -> begin
    (η1, )= unpack_parameters(Rayleigh, η)
    SA[inv(η1^2);;]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Rayleigh}) = (θ) -> begin
    (σ, ) = unpack_parameters(Rayleigh, θ)
    return 2*log(σ)
end

getfisherinformation(::MeanParametersSpace, ::Type{Rayleigh}) = (θ) -> begin
    (σ,  ) = unpack_parameters(Rayleigh, θ)
    return SA[4 / σ^2;;]
end

































# pack_naturalparameters(dist::Rayleigh) = [(-1/2) / first(params(dist))^2]
# function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:Rayleigh})
#     η = getnaturalparameters(ef)
#     @inbounds η1 = η[1]
#     return (η1,)
# end

# isproper(ef::ExponentialFamilyDistribution{Rayleigh}) = first(unpack_naturalparameters(ef)) < 0

# Base.convert(::Type{ExponentialFamilyDistribution}, dist::Rayleigh) =
#     ExponentialFamilyDistribution(Rayleigh, pack_naturalparameters(dist))

# function Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{Rayleigh})
#     (η,) = unpack_naturalparameters(ef)
#     return Rayleigh(sqrt(-1 / (2η)))
# end

# check_valid_natural(::Type{<:Rayleigh}, v) = length(v) === 1

# logpartition(ef::ExponentialFamilyDistribution{Rayleigh}) = -log(-2 * first(unpack_naturalparameters(ef)))

# fisherinformation(dist::Rayleigh) = SA[4 / scale(dist)^2;;]

# fisherinformation(ef::ExponentialFamilyDistribution{Rayleigh}) = SA[inv(first(unpack_naturalparameters(ef))^2);;]

# support(::ExponentialFamilyDistribution{Rayleigh}) = ClosedInterval{Real}(0, Inf)

# basemeasureconstant(::ExponentialFamilyDistribution{Rayleigh}) = NonConstantBaseMeasure()
# basemeasureconstant(::Type{<:Rayleigh}) = NonConstantBaseMeasure()

# sufficientstatistics(ef::Union{<:ExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}) =
#     (x) -> sufficientstatistics(ef, x)
# function sufficientstatistics(::Union{<:ExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x::Real)
#     return SA[x^2]
# end
# basemeasure(ef::Union{<:ExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}) = (x) -> basemeasure(ef, x)
# function basemeasure(::Union{<:ExponentialFamilyDistribution{Rayleigh}, <:Rayleigh}, x::Real)
#     return x
# end
