export ExponentialFamilyDistribution

using Distributions, LinearAlgebra, StaticArrays

import Random: rand

struct Safe end
struct Unsafe end

"""
    ExponentialFamilyDistribution(::Type{T}, naturalparameters, [ conditioner, basemeasure, sufficientstatistics, logpartition, support, supportcheck ])

`ExponentialFamilyDistribution` structure represents a generic exponential family distribution in natural parameterization.
Methods defined are `basemeasure` ,`sufficientstatistics`,  `naturalparameters`, `logpartition` and `support`.

- `getbasemeasure` returns a positive a valued function. 
- `getsufficientstatistics` returns an iterable of functions such as [x, x^2] or [x, logx].
- `getnaturalparameters` returns an iterable holding the values of the natural parameters. 
- `getlogpartition` return a function that depends on the naturalparameters and it ensures that the distribution is normalized to 1. 
- `support` returns the set that the distribution is defined over. Could be real numbers, positive integers, 3d cube etc. Use the `insupport` to check if a values is in support.

See also: [`getbasemeasure`](@ref), [`getsufficientstatistics`](@ref), [`getnaturalparameters`](@ref), [`getlogpartition`](@ref)
"""
struct ExponentialFamilyDistribution{T, P, C, H, S, Z, A, B}
    naturalparameters::P
    conditioner::C
    basemeasure::H
    sufficientstatistics::S
    logpartition::Z
    support::A
    supportcheck::B
    ExponentialFamilyDistribution(::Type{T},naturalparameters::P,conditioner::C, basemeasure::H, sufficientstatistics::S,
        logpartition::Z, support::A = nothing, supportcheck::B = Unsafe()) where {T, P, C, H, S, Z, A, B} = begin
            new{T, P, C, H, S, Z, A, B}(naturalparameters, conditioner, basemeasure, sufficientstatistics, logpartition, support, supportcheck)
    end
end

function ExponentialFamilyDistribution(::Type{T}, naturalparameters::AbstractVector, conditioner = nothing) where {T <: Distribution}
    @assert check_valid_natural(T, naturalparameters) == true "Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T)"
    @assert check_valid_conditioner(T, conditioner) "$(conditioner) is not a valid conditioner for distribution $(T) or 'check_valid_conditioner' function is not implemented!"
    return ExponentialFamilyDistribution(T, naturalparameters, conditioner, nothing, nothing, nothing, nothing, nothing)
end

getlogpartition(ef::ExponentialFamilyDistribution) = getlogpartition(ef.logpartition, ef)
getlogpartition(::Nothing, ef::ExponentialFamilyDistribution)  = logpartition(ef)
getlogpartition(something, ::ExponentialFamilyDistribution) = something

getbasemeasure(ef::ExponentialFamilyDistribution) = getbasemeasure(ef.basemeasure, ef)
getbasemeasure(::Nothing, ef::ExponentialFamilyDistribution) = basemeasure(ef)
getbasemeasure(something, ::ExponentialFamilyDistribution) = something

getsufficientstatistics(ef::ExponentialFamilyDistribution) = getsufficientstatistics(ef.sufficientstatistics, ef)
getsufficientstatistics(::Nothing, ef::ExponentialFamilyDistribution) = sufficientstatistics(ef)
getsufficientstatistics(something, ::ExponentialFamilyDistribution) = something

getsupport(ef::ExponentialFamilyDistribution) = getsupport(ef.support,ef)
getsupport(::Nothing, ef::ExponentialFamilyDistribution) = support(ef)
getsupport(something, ::ExponentialFamilyDistribution) = something

getconditioner(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.conditioner
getnaturalparameters(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.naturalparameters

struct ConstantBaseMeasure end 
struct NonConstantBaseMeasure end
basemeasureconstant(::ExponentialFamilyDistribution) = ConstantBaseMeasure()

Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T}, x) where {T <: Distribution}= logpdf(exponentialfamily,x, basemeasureconstant(exponentialfamily))

function Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T, H, S, P,C, Z, A, B}, x, ::ConstantBaseMeasure) where 
    {T <: Distribution,H,S,P,C,Z,A,B}
    @assert insupport(exponentialfamily, x)
    η = getnaturalparameters(exponentialfamily)
    statistics = getsufficientstatistics(exponentialfamily)(x)
    basemeasure = getbasemeasure(exponentialfamily)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure) + dot(η, statistics) - logpartition
end

function Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T, H, S, P,C, Z, A, B}, x, ::NonConstantBaseMeasure) where 
    {T <: Distribution,H,S,P,C,Z,A,B}
    @assert insupport(exponentialfamily, x)
    η = getnaturalparameters(exponentialfamily)
    statistics = getsufficientstatistics(exponentialfamily)(x)
    basemeasure = getbasemeasure(exponentialfamily)(x)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure) + dot(η, statistics) - logpartition
end

function Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T, H, S, P,C, Z, A, B}, x) where 
    {T <: VariateForm,H,S,P,C,Z,A,B}
    @assert insupport(exponentialfamily, x)
    η = getnaturalparameters(exponentialfamily)
    statistics = getsufficientstatistics(exponentialfamily)(x)
    basemeasure = getbasemeasure(exponentialfamily)(x)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure) + dot(η, statistics) - logpartition(η)
end

Distributions.pdf(exponentialfamily::ExponentialFamilyDistribution, x) = exp(logpdf(exponentialfamily, x)) 
Distributions.cdf(exponentialfamily::ExponentialFamilyDistribution, x) =
    Distributions.cdf(Base.convert(Distribution, exponentialfamily), x)

insupport(ef::ExponentialFamilyDistribution, x) = insupport(ef.supportcheck, ef, x)

insupport(::Safe, ef, x) = x ∈ support(ef)
insupport(::Unsafe, ef, x) = true
insupport(::Nothing, ef, x) = true


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
Everywhere in the package, we stick to a convention that we represent exponential family distributions in the following form:

``f_X(x\\mid\\theta) = h(x)\\,\\exp\\!\\bigl[\\,\\eta(\\theta) \\cdot T(x) - A(\\theta)\\,\\bigr]``.

So the `logpartition` sign should align with this form.
"""
function logpartition end

function basemeasure end
function sufficientstatistics end

"""
Fisher information
"""
function fisherinformation end

"""
Reconstructs an AbstractArray from a flattened Vector of values ηvec so that its shape matches that of the AbstractArray η.

If a unique element of η corresponds to a scalar value, the scalar is assigned directly to the corresponding index of η. If the unique element of η is a
non-scalar value, the function reshapes the appropriate slice of ηvec to match the shape of that element and assigns it to the corresponding indices of η.

Use the optional start argument to specify the beginning index when flattening η.

This function is useful for converting vectorized parameters into an appropriate size of natural parameters for a particular distribution.

Arguments
===========
•  η: Mutable AbstractArray to store the reconstructed values. The size and shape of η should match the desired size and shape of the reconstructed AbstractArray.
•  ηvec: A Vector containing the flattened values of the target AbstractArray.
•  start (optional): An integer argument used to set the starting index of ηvec.
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
rand(ef::ExponentialFamilyDistribution{T}) where {T <: Distribution} = rand(convert(T,ef))