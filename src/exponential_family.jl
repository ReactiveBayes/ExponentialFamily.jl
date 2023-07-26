using Distributions, LinearAlgebra, StaticArrays
import Random: rand

"""
    ExponentialFamilyDistribution{T, H, S, P, Z, A}

    `ExponentialFamilyDistribution` structure represents a generic exponential family distribution in natural parameterization.
    Its fields are `basemeasure` ,`sufficientstatistics`,  `naturalparameters`, `logpartition` and `support`. 
    `basemeasure` is a positive a valued function. `sufficientstatistics` is a vector of functions such as [x, x^2] or [x, logx].
    `naturalparameters` is an `AbstractArray` holding the values of the natural parameters. `logpartition` is a function that depends 
    on the naturalparameters and it ensures that the distribution is normalized to 1. `support` is the set that the distribution is 
    defined over. Could be real numbers, positive integers, 3d cube etc. 
"""
struct Safe end
struct Unsafe end

struct ExponentialFamilyDistribution{T, H, S, P,C, Z, A, B}
    basemeasure::H
    sufficientstatistics::S
    naturalparameters::P
    conditioner::C
    logpartition::Z
    support::A
    supportcheck::B
    ExponentialFamilyDistribution(::Type{T}, basemeasure::H, sufficientstatistics::S,
        naturalparameters::P,conditioner::C, logpartition::Z, support::A = nothing, supportcheck::B = Unsafe()) where {T, H, S, P, C, Z, A, B} = begin
            new{T, H, S, P,C, Z, A, B}(basemeasure, sufficientstatistics, naturalparameters,conditioner, logpartition, support, supportcheck)
    end
end


struct ConstantBaseMeasure end 
struct NonConstantBaseMeasure end

## These constructors create ExponentialFamily with known log partition
function ExponentialFamilyDistribution(::Type{T}, naturalparameters::Vector{P}) where {T <:Distribution, P}
    @assert check_valid_natural(T, naturalparameters) == true "Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T)"
    lp = logpartition(T, naturalparameters)
    return ExponentialFamilyDistribution(T, basemeasure(T), sufficientstatistics(T), naturalparameters, nothing,lp, support(T))
end

function ExponentialFamilyDistribution(::Type{T}, naturalparameters::Vector{P}, conditioner::C) where {T <: Distribution, P, C}
    @assert check_valid_natural(T, naturalparameters) == true "Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T)"
    @assert check_valid_conditioner(T, conditioner) "$(conditioner) is not a valid conditioner for distribution $(T) or 'check_valid_conditioner' function is not implemented!"
    lp = logpartition(T, naturalparameters)
    return ExponentialFamilyDistribution(T, basemeasure(T), sufficientstatistics(T), naturalparameters, conditioner, lp, support(T))
end

getnaturalparameters(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.naturalparameters
getlogpartition(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.logpartition
getbasemeasure(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.basemeasure
getsufficientstatistics(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.sufficientstatistics
getconditioner(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.conditioner
getsupport(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.support

basemeasureconstant(::ExponentialFamilyDistribution) = ConstantBaseMeasure()

Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T}, x) where {T <: Distribution}= logpdf(exponentialfamily,x, basemeasureconstant(exponentialfamily))

function Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T, H, S, P,C, Z, A, B}, x, ::ConstantBaseMeasure) where 
    {T <: Distribution,H,S,P,C,Z,A,B}
    @assert insupport(exponentialfamily,x)
    η = getnaturalparameters(exponentialfamily)
    statistics = sufficientstatistics(exponentialfamily,x)
    basemeasure = getbasemeasure(exponentialfamily)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure) + dot(η, statistics) - logpartition
end

function Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution{T, H, S, P,C, Z, A, B}, x, ::NonConstantBaseMeasure) where 
    {T <: Distribution,H,S,P,C,Z,A,B}
    @assert insupport(exponentialfamily,x)
    η = getnaturalparameters(exponentialfamily)
    statistics = sufficientstatistics(exponentialfamily, x)
    basemeasure = basemeasure(exponentialfamily, x)
    logpartition = getlogpartition(exponentialfamily)
    return log(basemeasure) + dot(η, statistics) - logpartition
end

Distributions.pdf(exponentialfamily::ExponentialFamilyDistribution, x) = exp(logpdf(exponentialfamily, x)) 
Distributions.cdf(exponentialfamily::ExponentialFamilyDistribution, x) =
    Distributions.cdf(Base.convert(Distribution, exponentialfamily), x)

function insupport(ef::ExponentialFamilyDistribution{T, H, S, P,C, Z, A, Safe}, x) where {T <: Distribution,H,S,P,C,Z,A}
    return x ∈ support(ef)
end

function insupport(::ExponentialFamilyDistribution{T, H, S, P,C, Z, A, Unsafe}, x) where {T <: Distribution,H,S,P,C,Z,A}
    return true
end



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