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
struct KnownExponentialFamilyDistribution{T, P, C}
    naturalparameters::P
    conditioner::C
    KnownExponentialFamilyDistribution(::Type{T}, naturalparameters::P, conditioner::C = nothing) where {T, P, C} =
        begin
            @assert check_valid_natural(T, naturalparameters) == true "Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T)"
            @assert check_valid_conditioner(T, conditioner) "$(conditioner) is not a valid conditioner for distribution $(T) or 'check_valid_conditioner' function is not implemented!"
            new{T, P, C}(naturalparameters, conditioner)
        end
end

variate_form(::P) where {P <: KnownExponentialFamilyDistribution}     = variate_form(P)
variate_form(::Type{KnownExponentialFamilyDistribution{T}}) where {T} = variate_form(T)
distributiontype(::KnownExponentialFamilyDistribution{T}) where {T}   = T

check_valid_conditioner(::Type{T}, conditioner) where {T} = conditioner === nothing

function check_valid_natural end

getnaturalparameters(exponentialfamily::KnownExponentialFamilyDistribution) = exponentialfamily.naturalparameters
getconditioner(exponentialfamily::KnownExponentialFamilyDistribution) = exponentialfamily.conditioner

Base.convert(::Type{T}, exponentialfamily::KnownExponentialFamilyDistribution) where {T <: Distribution} =
    Base.convert(T, Base.convert(Distribution, exponentialfamily))

Base.:(==)(left::KnownExponentialFamilyDistribution, right::KnownExponentialFamilyDistribution) =
    getnaturalparameters(left) == getnaturalparameters(right) && getconditioner(left) == getconditioner(right) &&
    distributiontype(left) == distributiontype(right)

Base.:(≈)(left::KnownExponentialFamilyDistribution, right::KnownExponentialFamilyDistribution) =
    getnaturalparameters(left) ≈ getnaturalparameters(right) && getconditioner(left) == getconditioner(right) &&
    distributiontype(left) == distributiontype(right)

Distributions.logpdf(exponentialfamily::KnownExponentialFamilyDistribution, x) =
    Distributions.logpdf(Base.convert(Distribution, exponentialfamily), x)
Distributions.pdf(exponentialfamily::KnownExponentialFamilyDistribution, x) =
    Distributions.pdf(Base.convert(Distribution, exponentialfamily), x)
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
