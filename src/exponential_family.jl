using Distributions, LinearAlgebra

"""
    ExponentialFamilyDistribution{T, H, S, P, Z, A}

    `ExponentialFamilyDistribution` structure represents a generic exponential family distribution in natural parameterization.
    Its fields are `basemeasure` ,`sufficientstatistics`,  `naturalparameters`, `logpartition` and `support`. 
    `basemeasure` is a positive a valued function. `sufficientstatistics` is a vector of functions such as [x, x^2] or [x, logx].
    `naturalparameters` is an `AbstractArray` holding the values of the natural parameters. `logpartition` is a function that depends 
    on the naturalparameters and it ensures that the distribution is normalized to 1. `support` is the set that the distribution is 
    defined over. Could be real numbers, positive integers, 3d cube etc. 
"""
struct ExponentialFamilyDistribution{T, H, S, P, Z, A}
    basemeasure::H
    sufficientstatistics::S
    naturalparameters::P
    logpartition::Z
    support::A
    ExponentialFamilyDistribution(::Type{T}, basemeasure::H, sufficientstatistics::S,
        naturalparameters::P, logpartition::Z, support::A = nothing) where {T, H, S, P, Z, A} = begin
        new{T, H, S, P, Z, A}(basemeasure, sufficientstatistics, naturalparameters, logpartition, support)
    end
end

getnaturalparameters(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.naturalparameters
getlogpartition(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.logpartition
getbasemeasure(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.basemeasure
getsufficientstatistics(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.sufficientstatistics
Distributions.support(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.support

variate_form(::P) where {P <: ExponentialFamilyDistribution}     = variate_form(P)
variate_form(::Type{ExponentialFamilyDistribution{T}}) where {T} = T

Base.:(==)(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution) =
    getnaturalparameters(left) == getnaturalparameters(right) && getlogpartition(left) == getlogpartition(right) &&
    getbasemeasure(left) == getbasemeasure(right) && variate_form(left) == variate_form(right) &&
    getsufficientstatistics(left) == getsufficientstatistics(right) &&
    Distributions.support(left) == Distributions.support(right)

Base.:(≈)(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution) =
    getnaturalparameters(left) ≈ getnaturalparameters(right) && getlogpartition(left) == getlogpartition(right) &&
    getbasemeasure(left) == getbasemeasure(right) && variate_form(left) == variate_form(right) &&
    getsufficientstatistics(left) == getsufficientstatistics(right) &&
    Distributions.support(left) == Distributions.support(right)

function Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution, x)
    η = getnaturalparameters(exponentialfamily)
    statistics = getsufficientstatistics(exponentialfamily)
    basemeasure = getbasemeasure(exponentialfamily)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure(x)) + dot(η, statistics(x)) - logpartition(η)
end

Distributions.pdf(exponentialfamily::ExponentialFamilyDistribution, x) = exp(logpdf(exponentialfamily, x))

"""
    KnownExponentialFamilyDistribution{T, P, C}

    `KnownExponentialFamilyDistribution` structure represents an exponential family distribution whose lognormalization is known.
    It is parameterized by a `Distribution` type from Distributions.jl. Its fields are `naturalparameters` holding a vector of natural parameters
    and `conditioner` that holds a constant parameter of `Distribution` such that it can be represented by an exponential family. For example,
    Gaussian, Gamma, etc. do not need a conditioner field because they are in exponential family unconditionally. However, Binomial and Multinomial 
    distributions are in exponential family give a parameter is fixed.

"""
struct KnownExponentialFamilyDistribution{T, P, C, S}
    naturalparameters::P
    conditioner::C
    support::S
    KnownExponentialFamilyDistribution(
        ::Type{T},
        naturalparameters::P,
        conditioner::C,
        support::S,
    ) where {T, P, C, S} =
        begin
            @assert check_valid_natural(T, naturalparameters) == true "Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T)"
            @assert check_valid_conditioner(T, conditioner) "$(conditioner) is not a valid conditioner for distribution $(T) or 'check_valid_conditioner' function is not implemented!"
            new{T, P, C, S}(naturalparameters, conditioner, support)
        end
end

function KnownExponentialFamilyDistribution(::Type{T}, naturalparameters::P) where {T, P}
    return KnownExponentialFamilyDistribution(T, naturalparameters, nothing, Safe())
end

function KnownExponentialFamilyDistribution(::Type{T}, naturalparameters::P, conditioner::C) where {T, P, C}
    return KnownExponentialFamilyDistribution(T, naturalparameters, conditioner, Safe())
end

struct Safe end

struct Unsafe end

function insupport(::KnownExponentialFamilyDistribution{T, P, C, Unsafe}, x) where {T, P, C}
    return true
end

function insupport(ef::KnownExponentialFamilyDistribution{T, P, C, Safe}, x) where {T, P, C}
    return x ∈ support(ef)
end

variate_form(::P) where {P <: KnownExponentialFamilyDistribution}     = variate_form(P)
variate_form(::Type{KnownExponentialFamilyDistribution{T}}) where {T} = variate_form(T)
distributiontype(::KnownExponentialFamilyDistribution{T}) where {T}   = T

check_valid_conditioner(::Type{T}, conditioner) where {T} = conditioner === nothing

function check_valid_natural end

function getnaturalparameters(exponentialfamily::KnownExponentialFamilyDistribution; vector = false)
    if vector == false
        return exponentialfamily.naturalparameters
    else
        return vcat(as_vec.(exponentialfamily.naturalparameters)...)
    end
end
getconditioner(exponentialfamily::KnownExponentialFamilyDistribution) = exponentialfamily.conditioner

Base.convert(::Type{T}, exponentialfamily::KnownExponentialFamilyDistribution) where {T <: Distribution} =
    Base.convert(T, Base.convert(Distribution, exponentialfamily))

Base.:(==)(left::KnownExponentialFamilyDistribution, right::KnownExponentialFamilyDistribution) =
    getnaturalparameters(left) == getnaturalparameters(right) && getconditioner(left) == getconditioner(right) &&
    distributiontype(left) == distributiontype(right)

Base.:(≈)(left::KnownExponentialFamilyDistribution, right::KnownExponentialFamilyDistribution) =
    getnaturalparameters(left) ≈ getnaturalparameters(right) && getconditioner(left) == getconditioner(right) &&
    distributiontype(left) == distributiontype(right)

function Distributions.logpdf(exponentialfamily::KnownExponentialFamilyDistribution, x)
    base_measure = log(basemeasure(exponentialfamily, x))
    natural_parameters = getnaturalparameters(exponentialfamily, vector = true)
    statistics = sufficientstatistics(exponentialfamily, x, vector = true)
    dot_product = dot(natural_parameters, statistics)

    return base_measure + dot_product - logpartition(exponentialfamily)
end
Distributions.pdf(exponentialfamily::KnownExponentialFamilyDistribution, x) = exp(logpdf(exponentialfamily, x))
Distributions.cdf(exponentialfamily::KnownExponentialFamilyDistribution, x) =
    Distributions.cdf(Base.convert(Distribution, exponentialfamily), x)

"""
Everywhere in the package, we stick to a convention that we represent exponential family distributions in the following form:

``f_X(x\\mid\\theta) = h(x)\\,\\exp\\!\\bigl[\\,\\eta(\\theta) \\cdot T(x) - A(\\theta)\\,\\bigr]``.

So the `logpartition` sign should align with this form.
"""
function logpartition end

function basemeasure end
function sufficientstatistics end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{T}, x; vector = false) where {T}
    if vector == false
        return sufficientstatistics(ef, x)
    else
        return vcat(as_vec.(sufficientstatistics(ef, x))...)
    end
end

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
