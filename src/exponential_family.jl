export ExponentialFamilyDistribution

export ExponentialFamilyDistribution, ExponentialFamilyDistributionAttributes, getnaturalparameters, getattributes
export MeanToNatural, NaturalToMean, MeanParametersSpace, NaturalParametersSpace
export getbasemeasure, getsufficientstatistics, getlogpartition, getfisherinformation, getsupport
export basemeasure, sufficientstatistics, logpartition, fisherinformation, insupport
export isbasemeasureconstant, ConstantBaseMeasure, NonConstantBaseMeasure

using LoopVectorization
using Distributions, LinearAlgebra, StaticArrays, Random

import Base: map

"""
    MeanToNatural(::Type{T})
 
Return the transformation function that maps the parameters in the mean parameters space to the natural parameters space for a distribution of type `T`.

See also: [`NaturalToMean`](@ref)
"""
struct MeanToNatural{T} end

MeanToNatural(::Type{T}) where T = MeanToNatural{T}()

"""
    NaturalToMean(::Type{T})

Return the transformation function that maps the parameters in the natural parameters space to the mean parameters space for a distribution of type `T`.

See also: [`MeanToNatural`](@ref)
"""
struct NaturalToMean{T} end

NaturalToMean(::Type{T}) where T = NaturalToMean{T}()

"""
    MeanParametersSpace

Specifies the mean parameters space `θ` as the desired parameters space.
Some functions (such as `logpartition` or `fisherinformation`) accept an additional `space` parameter to disambiguate the desired parameters space. 

See also: [`NaturalParametersSpace`](@ref)
"""
struct MeanParametersSpace end

"""
    NaturalParametersSpace

Specifies the natural parameters space `η` as the desired parameters space.
Some functions (such as `logpartition` or `fisherinformation`) accept an additional `space` parameter to disambiguate the desired parameters space. 

See also: [`MeanParametersSpace`](@ref)
"""
struct NaturalParametersSpace end

"""
    getbasemeasure(::ExponentialFamilyDistribution)

Returns the base measure function of the exponential family distribution.
"""
function getbasemeasure end

"""
    getsufficientstatistics(::ExponentialFamilyDistribution)

Returns the list of sufficient statistics of the exponential family distribution.
"""
function getsufficientstatistics end

"""
    getlogpartition(::ExponentialFamilyDistribution)

Returns the log partition function of the exponential family distribution.
"""
function getlogpartition end

"""
    getfisherinformation(::ExponentialFamilyDistribution)

Returns the function that computes the fisher information matrix of the exponential family distribution.
"""
function getfisherinformation end

"""
    getsupport(distribution_or_type)

Returns the support of the exponential family distribution.
"""
function getsupport end

"""
    insupport(support, value)

Checks if the given `value` belongs to the `support`.
"""
function insupport end

"""
    ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, support)

A structure to represent the attributes of an exponential family member.

# Fields
- `basemeasure::B`: The basemeasure of the exponential family member.
- `sufficientstatistics::S`: The sufficient statistics of the exponential family member.
- `logpartition::L`: The log-partition (cumulant) of the exponential family member.
- `support::P`: The support of the exponential family member.

See also: [`ExponentialFamilyDistribution`](@ref)
"""
struct ExponentialFamilyDistributionAttributes{B, S, L, P}
    basemeasure::B
    sufficientstatistics::S
    logpartition::L
    support::P
end

getbasemeasure(attributes::ExponentialFamilyDistributionAttributes) = attributes.basemeasure
getsufficientstatistics(attributes::ExponentialFamilyDistributionAttributes) = attributes.sufficientstatistics
getlogpartition(attributes::ExponentialFamilyDistributionAttributes) = attributes.logpartition
getsupport(attributes::ExponentialFamilyDistributionAttributes) = attributes.support

insupport(attributes::ExponentialFamilyDistributionAttributes, value) = insupport(getsupport(attributes), value)

"""
    ExponentialFamilyDistribution(::Type{T}, naturalparameters, conditioner, attributes)

`ExponentialFamilyDistribution` structure represents a generic exponential family distribution in natural parameterization.
Type `T` can be either a distribution type (e.g. from the `Distributions.jl` package) or a variate type (e.g. `Univariate`).

- `getattributes` returns either `nothing` or `ExponentialFamilyDistributionAttributes`.
- `getbasemeasure` returns a positive a valued function. 
- `getsufficientstatistics` returns an iterable of functions such as [x, x^2] or [x, logx].
- `getnaturalparameters` returns an iterable holding the values of the natural parameters. 
- `getlogpartition` return a function that depends on the naturalparameters and it ensures that the distribution is normalized to 1. 
- `getsupport` returns the set that the distribution is defined over. Could be real numbers, positive integers, 3d cube etc. Use the `insupport` to check if a values is in support.

!!! note
    The `attributes` can be `nothing`. In which case the package will try to derive the corresponding attributes from the type `T`.

See also: [`getbasemeasure`](@ref), [`getsufficientstatistics`](@ref), [`getnaturalparameters`](@ref), [`getlogpartition`](@ref), [`getsupport`](@ref)
"""
struct ExponentialFamilyDistribution{T, P, C, A}
    naturalparameters::P
    conditioner::C
    attributes::A

    ExponentialFamilyDistribution(
        ::Type{T},
        naturalparameters::P,
        conditioner::C = nothing,
        attributes::A = nothing
    ) where {T, P, C, A} = begin
        new{T, P, C, A}(
            naturalparameters,
            conditioner,
            attributes
        )
    end
end

function ExponentialFamilyDistribution(
    ::Type{T},
    naturalparameters::P,
    conditioner = nothing
) where {T <: Distribution, P}
    if !isproper(NaturalParametersSpace(), T, naturalparameters)
        error("Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T).")
    end
    if !check_valid_conditioner(T, conditioner)
        error("$(conditioner) is not a valid conditioner for distribution $(T).")
    end
    return ExponentialFamilyDistribution(T, naturalparameters, conditioner, nothing)
end


"""
    isproper(::ExponentialFamilyDistribution)

Checks if the object of type `ExponentialFamilyDistribution` is a proper distribution.
"""
isproper(::ExponentialFamilyDistribution) = true

"""
    getnaturalparameters(::ExponentialFamilyDistribution)

Get the natural parameters of the exponential family distribution.
"""
getnaturalparameters(ef::ExponentialFamilyDistribution) = ef.naturalparameters

"""
    getconditioner(::ExponentialFamilyDistribution)

Returns either the conditioner of the exponential family distribution or `nothing`. `conditioner` is a fixed parameter that is used to ensure that the distribution belongs to the exponential family.
"""
getconditioner(ef::ExponentialFamilyDistribution) = ef.conditioner

"""
    getattributes(::ExponentialFamilyDistribution)

Returns iether the attributes of the exponential family member or `nothing`. 

See also: [`ExponentialFamilyDistributionAttributes`](@ref)
"""
getattributes(ef::ExponentialFamilyDistribution) = ef.attributes

"""
    basemeasure(::ExponentialFamilyDistribution, x)

Returns the computed value of `basemeasure` of the exponential family distribution at the point `x`.

See also: [`getbasemeasure`](@ref)
"""
function basemeasure(ef::ExponentialFamilyDistribution, x)
    return getbasemeasure(ef)(x)
end

"""
    sufficientstatistics(::ExponentialFamilyDistribution)

Returns the computed values of `sufficientstatistics` of the exponential family distribution at the point `x`.
"""
function sufficientstatistics(ef::ExponentialFamilyDistribution, x)
    return map(f -> f(x), getsufficientstatistics(ef))
end

"""
    logpartition(::ExponentialFamilyDistribution, η)

Return the computed value of `logpartition` of the exponential family distribution at the point `η`.
By default `η = getnaturalparameters(ef)`.

See also: [`getlogpartition`](@ref)
"""
function logpartition(ef::ExponentialFamilyDistribution, η = getnaturalparameters(ef))
    return getlogpartition(ef)(η)
end

"""
    fisherinformation(distribution, η)

Return the computed value of `fisherinformation` of the exponential family distribution at the point `η`
By default `η = getnaturalparameters(ef)`.

See also: [`getfisherinformation`](@ref)
"""
function fisherinformation(ef::ExponentialFamilyDistribution, η = getnaturalparameters(ef)) 
    return getfisherinformation(ef)(η)
end

getbasemeasure(ef::ExponentialFamilyDistribution) = getbasemeasure(ef.attributes, ef)
getbasemeasure(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getbasemeasure(T)
getbasemeasure(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getbasemeasure(attributes)

getsufficientstatistics(ef::ExponentialFamilyDistribution) = getsufficientstatistics(ef.attributes, ef)
getsufficientstatistics(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getsufficientstatistics(T)
getsufficientstatistics(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getsufficientstatistics(attributes)

getlogpartition(ef::ExponentialFamilyDistribution) = getlogpartition(ef.attributes, ef)
getlogpartition(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getlogpartition(T)
getlogpartition(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getlogpartition(attributes)

getfisherinformation(ef::ExponentialFamilyDistribution) = getfisherinformation(ef.attributes, ef)
getfisherinformation(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getfisherinformation(T)
getfisherinformation(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    error("TODO: not implemented. Should we call ForwardDiff here?")

getsupport(ef::ExponentialFamilyDistribution) = getsupport(ef.attributes, ef)
getsupport(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getsupport(T)
getsupport(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getsupport(attributes)

insupport(ef::ExponentialFamilyDistribution, value) = insupport(getsupport(ef), value)

# For all `<:Distribution` the `support` function should be defined
getsupport(::Type{T}) where {T <: Distribution} = Distributions.support(T)

# Convenient mappings from a vectorized form to a vectorized form for distributions
(transformation::NaturalToMean{T})(v::AbstractVector) where { T <: Distribution } = pack_parameters(T, map(transformation, unpack_parameters(T, v)))
(transformation::MeanToNatural{T})(v::AbstractVector) where { T <: Distribution } = pack_parameters(T, map(transformation, unpack_parameters(T, v)))

"""
    isproper([ space = NaturalParametersSpace() ], ::Type{T}, parameters) where { T <: Distribution }

A specific verion of `isproper` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
Optionally, accepts the `space` parameter, which defines the parameters space.

See also: [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref)
"""
isproper(::Type{T}, parameters) where { T <: Distribution } = isproper(NaturalParametersSpace(), T, parameters)

isproper(ef::ExponentialFamilyDistribution{T}) where { T <: Distribution } = isproper(NaturalParametersSpace(), T, getnaturalparameters(ef))

"""
    getbasemeasure(::Type{T}) where { T <: Distribution }

A specific verion of `getbasemeasure` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
"""
getbasemeasure(::Type{T}) where { T <: Distribution }

"""
    getsufficientstatistics(::Type{T}) where { T <: Distribution }

A specific verion of `getsufficientstatistics` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
"""
getsufficientstatistics(::Type{T}) where { T <: Distribution }

"""
    getlogpartition([ space = NaturalParametersSpace() ], ::Type{T}) where { T <: Distribution }

A specific verion of `getlogpartition` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
Optionally, accepts the `space` parameter, which defines the parameters space.

See also: [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref)
"""
getlogpartition(::Type{T}) where { T <: Distribution } = getlogpartition(NaturalParametersSpace(), T)

"""
    getfisherinformation([ space = NaturalParametersSpace() ], ::Type{T}) where { T <: Distribution }

A specific verion of `getfisherinformation` defined particularly for distribution types from `Distributions.jl` package.
Does not require an instance of the `ExponentialFamilyDistribution` and can be called directly with a specific distribution type instead.
Optionally, accepts the `space` parameter, which defines the parameters space.

See also: [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref)
"""
getfisherinformation(::Type{T}) where { T <: Distribution } = getfisherinformation(NaturalParametersSpace(), T)

"""
A trait object representing that the base measure is constant.
"""
struct ConstantBaseMeasure end

"""
A trait object representing that the base measure is **not** constant.
"""
struct NonConstantBaseMeasure end

"""
    isbasemeasureconstant(something)

Returns either `NonConstantBaseMeasure()` or `ConstantBaseMeasure()` depending on if the base measure is a constant with respect to the natural parameters of `something` or not.
By default the package assumes that any base measure in a form of the `Function` is not a constant.
It, however, is not true for basemeasure that simply return a constant. In such cases the `isbasemeasureconstant` must have a specific method.

See also: [`getbasemeasure`](@ref), [`basemeasure`](@ref)
"""
function isbasemeasureconstant end

isbasemeasureconstant(ef::ExponentialFamilyDistribution) = isbasemeasureconstant(getbasemeasure(ef))
isbasemeasureconstant(::ExponentialFamilyDistribution{T}) where {T <: Distribution} = isbasemeasureconstant(T)

# For safety, by default, we assume that any base measure function is not a constant
# This is not always the case, e.g. when a function is (η) -> 1
# In such cases the `isbasemeasureconstant` must have a specific method
isbasemeasureconstant(::Function) = NonConstantBaseMeasure()

"""
    logpdf(ef::ExponentialFamilyDistribution, x)

Evaluates and returns the log-density of the exponential family distribution for the input `x`.
"""
function Distributions.logpdf(ef::ExponentialFamilyDistribution, x)
    # TODO: Think of what to do with this assert
    @assert insupport(ef, x)

    η = getnaturalparameters(ef)

    # Use `_` to avoid name collisions with the actual functions
    _statistics = sufficientstatistics(ef, x)
    _basemeasure = basemeasure(ef, x)
    _logpartition = logpartition(ef)

    return log(_basemeasure) + dot(η, _statistics) - _logpartition
end

"""
    pdf(ef::ExponentialFamilyDistribution, x)

Evaluates and returns the probability density function of the exponential family distribution for the input `x`.
"""
Distributions.pdf(ef::ExponentialFamilyDistribution, x) = exp(logpdf(ef, x))

"""
    cdf(ef::ExponentialFamilyDistribution, x)

Evaluates and returns the cumulative distribution function of the exponential family distribution for the input `x`.
"""
Distributions.cdf(ef::ExponentialFamilyDistribution{D}, x) where {D <: Distribution} =
    Distributions.cdf(Base.convert(Distribution, ef), x)

variate_form(::Type{<:ExponentialFamilyDistribution{D}}) where {D <: Distribution} = variate_form(D)
variate_form(::Type{<:ExponentialFamilyDistribution{V}}) where {V <: VariateForm} = V

value_support(::Type{<:ExponentialFamilyDistribution{D}}) where {D <: Distribution} = value_support(D)

# HM TODO (bvdmitri), how to define value support for an arbitrary `Univariate`, this is not possible currently
# value_support(::Type{<:ExponentialFamilyDistribution{D}}) where { D <: Distribution } = value_support(D)

distributiontype(::Type{<:ExponentialFamilyDistribution{T}}) where {T <: Distribution} = T

"""
    check_valid_conditioner(distribution_type, conditioner)

Checks if the `conditioner` holds a correct value for the `distribution_type`.
"""
check_valid_conditioner(T, conditioner) =
    error("The `$(conditioner)` is not a valid conditioner for a distribution of type `$(T)`.")

"""
    pack_parameters(::Type{T}, params::Tuple)

This function returns the parameters of the `distribution` in a vectorized (packed) form.
Optionally accepts the `space` of the parameters, which is equal to `NaturalParametersSpace`.

See also: [`NaturalParametersSpace`](@ref), [`MeanParametersSpace`](@ref)
"""
function pack_parameters end

"""
    unpack_parameters(::Type{T}, parameters)

This function "unpack" the vectorized form of the parameters in a tuple.
"""
function unpack_parameters end

unpack_parameters(ef::ExponentialFamilyDistribution{T}) where {T} = unpack_parameters(T, getnaturalparameters(ef))

Base.convert(::Type{T}, ef::ExponentialFamilyDistribution) where {T <: Distribution} =
    Base.convert(T, Base.convert(Distribution, ef))

# Generic convert from an `ExponentialFamilyDistribution{T}` to its corresponding type `T`
function Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{T}) where { T <: Distribution }
    tuple_of_η = unpack_parameters(ef)
    params = NaturalToMean(T)(tuple_of_η)
    return T(params...)
end

# Generic convert from a member of `Distributions` to its equivalent representation in `ExponentialFamilyDistribution`
function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Distribution)
    T = distribution_typename(dist)
    tuple_if_θ = params(dist)
    tuple_of_η = MeanToNatural(T)(tuple_if_θ)
    η = pack_parameters(T, tuple_of_η)
    return ExponentialFamilyDistribution(T, η)
end

function paramfloattype(ef::ExponentialFamilyDistribution)
    return deep_eltype(getnaturalparameters(ef))
end

function convert_paramfloattype(::Type{F}, ef::ExponentialFamilyDistribution{T}) where {F, T}
    return ExponentialFamilyDistribution(
        T,
        convert_paramfloattype(F, getnaturalparameters(ef)),
        getconditioner(ef),
        getattributes(ef)
    )
end

Distributions.mean(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = mean(convert(T, ef))
Distributions.var(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = var(convert(T, ef))
Distributions.std(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = std(convert(T, ef))
Distributions.cov(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = cov(convert(T, ef))

Random.rand(ef::ExponentialFamilyDistribution, args...) = rand(Random.default_rng(), ef, args...)
Random.rand!(ef::ExponentialFamilyDistribution, args...) = rand!(Random.default_rng(), ef, args...)

Random.rand(rng::AbstractRNG, ef::ExponentialFamilyDistribution{T}, args::Integer...) where {T <: Distribution} =
    rand(rng, convert(T, ef), args...)

Random.rand!(rng::AbstractRNG, ef::ExponentialFamilyDistribution{T}, container) where {T <: Distribution} =
    rand!(rng, convert(T, ef), container)

Base.isapprox(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution; kwargs...) = false
Base.:(==)(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution) = false

function Base.isapprox(
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T};
    kwargs...
) where {T}
    return getbasemeasure(left) == getbasemeasure(right) &&
           getsufficientstatistics(left) == getsufficientstatistics(right) &&
           getlogpartition(left) == getlogpartition(right) && getsupport(left) == getsupport(right) &&
           getconditioner(left) == getconditioner(right) &&
           isapprox(getnaturalparameters(left), getnaturalparameters(right); kwargs...)
end

function Base.:(==)(left::ExponentialFamilyDistribution{T}, right::ExponentialFamilyDistribution{T}) where {T}
    return getbasemeasure(left) == getbasemeasure(right) &&
           getsufficientstatistics(left) == getsufficientstatistics(right) &&
           getlogpartition(left) == getlogpartition(right) && getsupport(left) == getsupport(right) &&
           getconditioner(left) == getconditioner(right) &&
           getnaturalparameters(left) == getnaturalparameters(right)
end

Base.similar(ef::ExponentialFamilyDistribution) = similar(ef, eltype(getnaturalparameters(ef)))

function Base.similar(ef::ExponentialFamilyDistribution{T}, ::Type{F}) where {T, F}
    return ExponentialFamilyDistribution(T, similar(getnaturalparameters(ef), F), getconditioner(ef), getattributes(ef))
end

vague(::Type{ExponentialFamilyDistribution{T}}, args...) where {T <: Distribution} =
    convert(ExponentialFamilyDistribution, vague(T, args...))

# We assume that we want to preserve the `ExponentialFamilyDistribution` when working with two `ExponentialFamilyDistribution`s
default_prod_rule(::Type{<:ExponentialFamilyDistribution}, ::Type{<:ExponentialFamilyDistribution}) =
    PreserveTypeProd(ExponentialFamilyDistribution)

function prod(::ClosedProd, left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution)
    return prod(PreserveTypeProd(ExponentialFamilyDistribution), left, right)
end

function prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T}
    # Se here we assume that if both left has the exact same base measure and this base measure is `ConstantBaseMeasure`
    # We assume that this code-path is static and should be const-folded in run-time (there are tests that check that this function does not allocated more than `similar(left)`)
    if isbasemeasureconstant(left) === ConstantBaseMeasure() &&
       isbasemeasureconstant(right) === ConstantBaseMeasure() && getbasemeasure(left) === getbasemeasure(right)
        if isnothing(getconditioner(left)) && isnothing(getconditioner(right))
            # Find the promoted float type of both natural parameters
            F = promote_type(eltype(getnaturalparameters(left)), eltype(getnaturalparameters(right)))
            # Create a suitable container for the in-place `prod!` operation, we use the `left` as a suitable candidate
            container = similar(left, F)
            return Base.prod!(container, left, right)
        end
    end
    error("Generic product of two exponential family members is not implemented.")
end

function Base.prod!(
    container::ExponentialFamilyDistribution{T},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T}
    # First check if we can actually simply sum-up the natural parameters
    # We assume that this code-path is static and should be const-folded in run-time (there are tests that check that this function does not allocate in this simple case)
    if isbasemeasureconstant(left) === ConstantBaseMeasure() &&
       isbasemeasureconstant(right) === ConstantBaseMeasure() && getbasemeasure(left) === getbasemeasure(right)
        if isnothing(getconditioner(left)) && isnothing(getconditioner(right))
            LoopVectorization.vmap!(
                +,
                getnaturalparameters(container),
                getnaturalparameters(left),
                getnaturalparameters(right)
            )
            return container
        end
    end
    # If the check fails, do not do un-safe operation and simply fallback to the `PreserveTypeProd(ExponentialFamilyDistribution)`
    return prod(PreserveTypeProd(ExponentialFamilyDistribution), left, right)
end
