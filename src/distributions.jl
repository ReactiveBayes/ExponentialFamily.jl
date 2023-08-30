export vague
export mean, median, mode, shape, scale, rate, var, std, cov, invcov, entropy, pdf, cdf, logpdf, logdetcov
export mean_cov,
    mean_var, mean_std, mean_invcov, mean_precision, weightedmean_cov, weightedmean_var, weightedmean_std,
    weightedmean_invcov, weightedmean_precision
export weightedmean, probvec, isproper

using Distributions, Random

import Distributions: mean, median, mode, shape, scale, rate, var, std, cov, invcov, entropy, pdf, logpdf, logdetcov, insupport
import Distributions:
    VariateForm, ValueSupport, Distribution, Univariate, Multivariate, Matrixvariate, variate_form, value_support

import Base: convert

# We assume that we want to preserve the `Distribution` when working with two `Distribution`s
prod(::ClosedProd, left::Distribution, right::Distribution) = prod(PreserveTypeProd(Distribution), left, right)

"""
    vague(distribution_type, [ dims... ])

`vague` function returns uninformative probability distribution of a given type and can be used to set an uninformative priors in a model.
"""
function vague end

mean_cov(something)       = (mean(something), cov(something))
mean_var(something)       = (mean(something), var(something))
mean_std(something)       = (mean(something), std(something))
mean_invcov(something)    = (mean(something), invcov(something))
mean_precision(something) = mean_invcov(something)

weightedmean_cov(something)       = (weightedmean(something), cov(something))
weightedmean_var(something)       = (weightedmean(something), var(something))
weightedmean_std(something)       = (weightedmean(something), std(something))
weightedmean_invcov(something)    = (weightedmean(something), invcov(something))
weightedmean_precision(something) = weightedmean_invcov(something)

function probvec end

function weightedmean end

# Julia does not really like expressions of the form
# map((e) -> convert(T, e), collection)
# because type `T` is inside lambda function
# https://github.com/JuliaLang/julia/issues/15276
# https://github.com/JuliaLang/julia/issues/47760
struct PromoteTypeConverter{T, C}
    convert::C
end

PromoteTypeConverter(::Type{T}, convert::C) where {T, C} = PromoteTypeConverter{T, C}(convert)

(converter::PromoteTypeConverter{T})(something) where {T} = converter.convert(T, something)

"""
    promote_variate_PromoteTypeConverter(::Type{ <: VariateForm }, distribution_type)

Promotes (if possible) a `distribution_type` to be of the specified variate form.
"""
function promote_variate_type end

"""
    promote_variate_type(::Type{D}, distribution_type) where { D <: Distribution }

Promotes (if possible) a `distribution_type` to be of the same variate form as `D`.
"""
promote_variate_type(::Type{D}, T) where {D <: Distribution} = promote_variate_type(variate_form(D), T)

"""
    paramfloattype(distribution)

Returns the underlying float type of distribution's parameters.

See also: [`ExponentialFamily.promote_paramfloattype`](@ref), [`ExponentialFamily.convert_paramfloattype`](@ref)
"""
paramfloattype(distribution::Distribution) = promote_type(map(deep_eltype, params(distribution))...)
paramfloattype(nt::NamedTuple) = promote_paramfloattype(values(nt))
paramfloattype(t::Tuple) = promote_paramfloattype(t...)

# `Bool` is the smallest possible type, should not play any role in the promotion
paramfloattype(::Nothing) = Bool

"""
    promote_paramfloattype(distributions...)

Promotes `paramfloattype` of the `distributions` to a single type. See also `promote_type`.

See also: [`ExponentialFamily.paramfloattype`](@ref), [`ExponentialFamily.convert_paramfloattype`](@ref)
"""
promote_paramfloattype(distributions...) = promote_type(map(paramfloattype, distributions)...)

"""
    convert_paramfloattype(::Type{T}, distribution)

Converts (if possible) the params float type of the `distribution` to be of type `T`.

See also: [`ExponentialFamily.paramfloattype`](@ref), [`ExponentialFamily.promote_paramfloattype`](@ref)
"""
convert_paramfloattype(::Type{T}, distribution::Distribution) where {T} =
    automatic_convert_paramfloattype(
        distribution_typewrapper(distribution),
        map(convert_paramfloattype(T), params(distribution))
    )
convert_paramfloattype(::Type{T}, collection::NamedTuple) where {T} = map(convert_paramfloattype(T), collection)
convert_paramfloattype(collection::NamedTuple) = convert_paramfloattype(paramfloattype(collection), collection)
convert_paramfloattype(::Type{T}) where {T} = PromoteTypeConverter(T, convert_paramfloattype)

# We attempt to automatically construct a new distribution with a desired paramfloattype
# This function assumes that the constructor `D(...)` accepts the same order of parameters as 
# returned from the `params` function. It is the case for distributions from `Distributions.jl`
automatic_convert_paramfloattype(::Type{D}, params) where {D <: Distribution} = D(params...)
automatic_convert_paramfloattype(::Type{D}, params) where {D} =
    error("Cannot automatically construct a distribution of type `$D` with params = $(params)")

"""
    convert_paramfloattype(::Type{T}, container)

Converts (if possible) the elements of the `container` to be of type `T`.
"""
convert_paramfloattype(::Type{T}, container::AbstractArray) where {T} = convert(AbstractArray{T}, container)
convert_paramfloattype(::Type{T}, number::Number) where {T} = convert(T, number)
convert_paramfloattype(::Type, ::Nothing) = nothing

"""
    sampletype(distribution)

Returns a type of the distribution. By default fallbacks to the `eltype`.

See also: [`ExponentialFamily.samplefloattype`](@ref), [`ExponentialFamily.promote_sampletype`](@ref), [`ExponentialFamily.promote_samplefloattype`](@ref)
"""
sampletype(distribution) = eltype(distribution)

sampletype(distribution::Distribution) = sampletype(variate_form(typeof(distribution)), distribution)
sampletype(::Type{Univariate}, distribution) = eltype(distribution)
sampletype(::Type{Multivariate}, distribution) = Vector{eltype(distribution)}
sampletype(::Type{Matrixvariate}, distribution) = Matrix{eltype(distribution)}

"""
    samplefloattype(distribution)

Returns a type of the distribution or the underlying float type in case if sample is `Multivariate` or `Matrixvariate`. 
By default fallbacks to the `deep_eltype(sampletype(distribution))`.

See also: [`ExponentialFamily.sampletype`](@ref), [`ExponentialFamily.promote_sampletype`](@ref), [`ExponentialFamily.promote_samplefloattype`](@ref)
"""
samplefloattype(distribution) = deep_eltype(sampletype(distribution))

"""
    promote_sampletype(distributions...)

Promotes `sampletype` of the `distributions` to a single type. See also `promote_type`.

See also: [`ExponentialFamily.sampletype`](@ref), [`ExponentialFamily.samplefloattype`](@ref), [`ExponentialFamily.promote_samplefloattype`](@ref)
"""
promote_sampletype(distributions...) = promote_type(map(sampletype, distributions)...)

"""
    promote_samplefloattype(distributions...)

Promotes `samplefloattype` of the `distributions` to a single type. See also `promote_type`.

See also: [`ExponentialFamily.sampletype`](@ref), [`ExponentialFamily.samplefloattype`](@ref), [`ExponentialFamily.promote_sampletype`](@ref)
"""
promote_samplefloattype(distributions...) = promote_type(map(samplefloattype, distributions)...)

"""
    logpdf_sample_optimized(distribution) 
    
`logpdf_sample_optimized` function takes as an input a `distribution` and returns corresponding optimized two versions 
for taking `logpdf()` and sampling with `rand!` respectively. By default returns the same distribution, but some distributions 
may override default behaviour for better efficiency.

# Example

```jldoctest
julia> d = vague(MvNormalMeanPrecision, 2)
MvNormalMeanPrecision(
μ: [0.0, 0.0]
Λ: [1.0e-12 0.0; 0.0 1.0e-12]
)


julia> ExponentialFamily.logpdf_sample_optimized(d)
(FullNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0e12 -0.0; -0.0 1.0e12]
)
, FullNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0e12 -0.0; -0.0 1.0e12]
)
)
```
"""
logpdf_sample_optimized(something) = (logpdf_optimized(something), sample_optimized(something))

logpdf_optimized(something) = something
sample_optimized(something) = something

"""
    FactorizedJoint

`FactorizedJoint` represents a joint distribution of independent random variables. 
Use `getindex()` function or square-brackets indexing to access the marginal distribution for individual variables.
"""
struct FactorizedJoint{T}
    multipliers::T
end

getmultipliers(joint::FactorizedJoint) = joint.multipliers

Base.@propagate_inbounds Base.getindex(joint::FactorizedJoint, i::Int) = getindex(getmultipliers(joint), i)

Base.length(joint::FactorizedJoint) = length(joint.multipliers)

function Base.isapprox(x::FactorizedJoint, y::FactorizedJoint; kwargs...)
    length(x) === length(y) &&
        all(tuple -> isapprox(tuple[1], tuple[2]; kwargs...), zip(getmultipliers(x), getmultipliers(y)))
end

Distributions.entropy(joint::FactorizedJoint) = mapreduce(entropy, +, getmultipliers(joint))

paramfloattype(joint::FactorizedJoint) = paramfloattype(getmultipliers(joint))

convert_paramfloattype(::Type{T}, joint::FactorizedJoint) where {T} =
    FactorizedJoint(map(e -> convert_paramfloattype(T, joint), getmultipliers(joint)))

## Utils

distribution_typewrapper(distribution) = generated_distribution_typewrapper(distribution)

# Returns a wrapper distribution for a `<:Distribution` type, this function uses internals of Julia 
# It is not ideal, but is fine for now, if Julia changes it internals such that does not work 
# We will need to write the `distribution_typewrapper` method for each support member of exponential family
# e.g. `distribution_typewrapper(::Bernoulli) = Bernoulli`
@generated function generated_distribution_typewrapper(distribution)
    return Base.typename(distribution).wrapper
end
