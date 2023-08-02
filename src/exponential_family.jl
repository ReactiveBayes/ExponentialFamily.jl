export ExponentialFamilyDistribution

using Distributions, LinearAlgebra, StaticArrays

import Random: rand

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
    getsupport(::ExponentialFamilyDistribution)

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
    if !check_valid_natural(T, naturalparameters)
        error("Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T).")
    end
    if !check_valid_conditioner(T, conditioner)
        error("$(conditioner) is not a valid conditioner for distribution $(T).")
    end
    return ExponentialFamilyDistribution(T, naturalparameters, conditioner, nothing)
end

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
    basemeasure(::ExponentialFamilyDistribution)

Returns the computed value of `basemeasure` of the exponential family distribution.

See also: [`getbasemeasure`](@ref)
"""
function basemeasure(ef::ExponentialFamilyDistribution)
    return getbasemeasure(ef)(getnaturalparameters(ef))
end

"""
    sufficientstatistics(::ExponentialFamilyDistribution)

Returns the computed values of `sufficientstatistics` of the exponential family distribution.
"""
function sufficientstatistics(ef::ExponentialFamilyDistribution)
    return let η = getnaturalparameters(ef)
        map(f -> f(η), getsufficientstatistics(ef))
    end
end

"""
    logpartition(::ExponentialFamilyDistribution)

Return the computed value of `logpartition` of the exponential family distribution.

See also: [`getlogpartition`](@ref)
"""
function logpartition(ef::ExponentialFamilyDistribution)
    return getlogpartition(ef)(getnaturalparameters(ef))
end

"""
    fisherinformation(::ExponentialFamilyDistribution)

Return the computed value of `fisherinformation` of the exponential family distribution.
"""
function fisherinformation end

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

getsupport(ef::ExponentialFamilyDistribution) = getsupport(ef.attributes, ef)
getsupport(::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getsupport(T)
getsupport(attributes::ExponentialFamilyDistributionAttributes, ::ExponentialFamilyDistribution) =
    getsupport(attributes)

insupport(ef::ExponentialFamilyDistribution, value) = insupport(getsupport(ef), value)

struct ConstantBaseMeasure end
struct NonConstantBaseMeasure end

basemeasureconstant(::ExponentialFamilyDistribution) = ConstantBaseMeasure()

"""
    Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T}, x) where {T <: Distribution}

Evaluate the log-density of the exponential family distribution for the input `x` with constant base measure.

# Arguments
- `exponentialfamily`: The exponential family distribution.
- `x`: The input value.

# Returns
- The log-density of `exponentialfamily` evaluated at `x` with constant base measure.

"""
Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T}, x) where {T <: Distribution} =
    logpdf(exponentialfamily, x, basemeasureconstant(exponentialfamily))

"""
    Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T, H, S, P, C, Z, A, B}, x, ::ConstantBaseMeasure)

Evaluate the log-density of the exponential family distribution for the input `x` with constant base measure.

# Arguments
- `exponentialfamily`: The exponential family distribution.
- `x`: The input value.
- `::ConstantBaseMeasure`: A marker for the constant base measure (used for dispatch).

# Returns
- The log-density of `exponentialfamily` evaluated at `x` with constant base measure.

"""
function Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution, x, _)
    @assert insupport(exponentialfamily, x)
    η = getnaturalparameters(exponentialfamily)
    statistics = getsufficientstatistics(exponentialfamily)(x)
    basemeasure = getbasemeasure(exponentialfamily)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure) + dot(η, statistics) - logpartition
end

"""
    Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T, H, S, P, C, Z, A, B}, x, ::NonConstantBaseMeasure)

Evaluate the log-density of the exponential family distribution for the input `x` with non-constant base measure.

# Arguments
- `exponentialfamily`: The exponential family distribution.
- `x`: The input value.
- `::NonConstantBaseMeasure`: A marker for the non-constant base measure (used for dispatch).

# Returns
- The log-density of `exponentialfamily` evaluated at `x` with non-constant base measure.

"""
function Distributions.logpdf(
    exponentialfamily::ExponentialFamilyDistribution{T},
    x,
    ::NonConstantBaseMeasure
) where
    {T <: Distribution}
    @assert insupport(exponentialfamily, x)
    η = getnaturalparameters(exponentialfamily)
    statistics = getsufficientstatistics(exponentialfamily)(x)
    basemeasure = getbasemeasure(exponentialfamily)(x)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure) + dot(η, statistics) - logpartition
end

"""
    Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T, H, S, P, C, Z, A, B}, x)

Evaluate the log-density of the exponential family distribution for the input `x` with constant base measure.

# Arguments
- `exponentialfamily`: The exponential family distribution.
- `x`: The input value.

# Returns
- The log-density of `exponentialfamily` evaluated at `x` with constant base measure.

"""
function Distributions.logpdf(
    exponentialfamily::ExponentialFamilyDistribution{T},
    x
) where
{T <: VariateForm}
    @assert insupport(exponentialfamily, x)
    η = getnaturalparameters(exponentialfamily)
    statistics = getsufficientstatistics(exponentialfamily)(x)
    basemeasure = getbasemeasure(exponentialfamily)(x)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure) + dot(η, statistics) - logpartition(η)
end

"""
    Distributions.pdf(exponentialfamily::ExponentialFamilyDistribution, x)

Evaluate the probability density function of the exponential family distribution for the input `x`.
"""
Distributions.pdf(exponentialfamily::ExponentialFamilyDistribution, x) = exp(logpdf(exponentialfamily, x))
"""
    Distributions.cdf(exponentialfamily::ExponentialFamilyDistribution, x)

Evaluate the cumulative distribution function of the exponential family distribution for the input `x`.
"""
Distributions.cdf(exponentialfamily::ExponentialFamilyDistribution, x) =
    Distributions.cdf(Base.convert(Distribution, exponentialfamily), x)

"""
    insupport(ef::ExponentialFamilyDistribution, x)

Check if the input `x` is in the support of the exponential family distribution `ef`.
"""
insupport(ef::ExponentialFamilyDistribution, x) = x ∈ getsupport(ef)

variate_form(::P) where {P <: ExponentialFamilyDistribution} = variate_form(P)
variate_form(::Type{<:ExponentialFamilyDistribution{T}}) where {T} = variate_form(T)
distributiontype(::ExponentialFamilyDistribution{T}) where {T} = T
distributiontype(::Type{<:ExponentialFamilyDistribution{T}}) where {T} = T
check_valid_conditioner(::Type{T}, conditioner) where {T} = conditioner === nothing

function check_valid_natural end

Base.convert(::Type{T}, exponentialfamily::ExponentialFamilyDistribution) where {T <: Distribution} =
    Base.convert(T, Base.convert(Distribution, exponentialfamily))

Base.:(==)(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution) =
    getnaturalparameters(left) == getnaturalparameters(right) && getconditioner(left) == getconditioner(right) &&
    distributiontype(left) == distributiontype(right)

Base.:(≈)(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution) =
    getnaturalparameters(left) ≈ getnaturalparameters(right) && getconditioner(left) == getconditioner(right) &&
    distributiontype(left) == distributiontype(right)

"""
    reconstruct_array!(η, ηef, ηvec; start = 1)

Reconstruct an `AbstractArray` from a flattened `Vector` of values `ηvec` so that its shape matches that of the `AbstractArray` `η`.

If a unique element of `η` corresponds to a scalar value, the scalar is assigned directly to the corresponding index of `η`. If the unique element of `η` is a non-scalar value, the function reshapes the appropriate slice of `ηvec` to match the shape of that element and assigns it to the corresponding indices of `η`.

Use the optional `start` argument to specify the beginning index when flattening `ηvec`.

This function is useful for converting vectorized parameters into an appropriate size of natural parameters for a particular distribution.

# Arguments
- `η`: Mutable `AbstractArray` to store the reconstructed values. The size and shape of `η` should match the desired size and shape of the reconstructed `AbstractArray`.
- `ηvec`: A `Vector` containing the flattened values of the target `AbstractArray`.
- `start` (optional): An integer argument used to set the starting index of `ηvec`.

"""
function reconstructargument!(η, ηef, ηvec; start = 1)
    # Check if η and ηef have compatible dimensions
    @assert length(η) == length(ηef) "η and ηef must have the same length"

    # Check if η and ηvec have compatible dimensions
    expected_size = sum([length(elem) for elem in ηef])
    @assert length(ηvec) == expected_size "Expected size of ηef $(expected_size), but the ηvec has length $(length(ηvec))"

    @inbounds for i in eachindex(η)
        stop = start + length(ηef[i]) - 1
        ind = start:stop
        if length(ηef[i]) == 1
            η[i] = first(ηvec[ind])
        else
            @views η[i] = reshape(ηvec[ind], size(ηef[i]))
        end
        start = stop + 1
    end
    return η
end

mean(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = mean(convert(T, ef))
var(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = var(convert(T, ef))
cov(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = cov(convert(T, ef))
rand(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = rand(convert(T, ef))

# We assume that we want to preserve the `ExponentialFamilyDistribution` when working with two `ExponentialFamilyDistribution`s
default_prod_rule(::Type{<:ExponentialFamilyDistribution}, ::Type{<:ExponentialFamilyDistribution}) =
    PreserveTypeProd(ExponentialFamilyDistribution)

# Case when both supertypes are of type `Distribution` and we have the `ClosedProd` for them
# The idea here is that converting from `ExponentialFamilyDistribution` to a `Distribution` should be free
# So we simply convert `EF` representation to the `Distribution` representation, call their closed product and convert back
# function prod(
#     left::ExponentialFamilyDistribution{D1},
#     right::ExponentialFamilyDistribution{D2}
# ) where {D1 <: Distribution, D2 <: Distribution}
#     error("This method should be generalized to accept the `ClosedProd` as its first argument. TODO.")
#     # Should be compiled out anyway
#     if default_prod_rule(D1, D2) === ClosedProd()
#         return convert(
#             ExponentialFamilyDistribution,
#             prod(ClosedProd(), convert(Distribution, left), convert(Distribution, left))
#         )
#     end
#     # We assume that we can always execute the `ClosedProd` for any ExponentialFamilyDistribution
#     return prod(ClosedProd(), left, right)
# end

# function prod(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution)
#     return prod(ClosedProd(), left, right)
# end

# Case when both `ExponentialFamilyDistribution` are of the same `Distribution` type 
# But for some reason we don't have the `ClosedProd` defined for them
# function prod(
#     ::ClosedProd,
#     left::ExponentialFamilyDistribution{T},
#     right::ExponentialFamilyDistribution{T}
# ) where {T <: Distribution}
#     # Here we need to check that the basemeasures are constants and that the conditioners are the same 
#     # only then we can sum-up the natural parameters, for now I leave this method as `not implemented`
#     # but it is definitely should be properly implemented
#     error("Not properly implemented")
#     # ExponentialFamilyDistribution(
#     #     T,
#     #     getnaturalparameters(left) + getnaturalparameters(right),
#     #     getconditioner(left)
#     # )
# end

# function prod(::ClosedProd, left::Distribution{T}, right::Distribution{T}) where {T}
#     error("This method should go away.")
#     # efleft = convert(ExponentialFamilyDistribution, left)
#     # efright = convert(ExponentialFamilyDistribution, right)
#     # return convert(Distribution, prod(efleft, efright))
# end
