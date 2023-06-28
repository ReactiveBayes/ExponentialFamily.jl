export Laplace
using Distributions
import Distributions: Laplace, params, logpdf
using DomainSets

vague(::Type{<:Laplace}) = Laplace(0.0, huge)

closed_prod_rule(::Type{<:Laplace}, ::Type{<:Laplace}) = ClosedProd()

function Base.prod(
    ::ClosedProd,
    ef_left::KnownExponentialFamilyDistribution{T},
    ef_right::KnownExponentialFamilyDistribution{T}
) where {T <: Laplace}
    (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
    (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
    if conditioner_left == conditioner_right
        return KnownExponentialFamilyDistribution(Laplace, η_left + η_right, conditioner_left)
    else
        basemeasure = (x) -> 1.0
        sufficientstatistics = (x) -> [abs(x - conditioner_left), abs(x - conditioner_right)]
        sorted_conditioner = sort([conditioner_left, conditioner_right])
        function logpartition(η)
            A1 = exp(η[1] * conditioner_left + η[2] * conditioner_right)
            A2 = exp(-η[1] * conditioner_left + η[2] * conditioner_right)
            A3 = exp(-η[1] * conditioner_left - η[2] * conditioner_right)
            B1 = (exp(sorted_conditioner[2] * (-η[1] - η[2])) - 1.0) / (-η[1] - η[2])
            B2 =
                (exp(sorted_conditioner[1] * (η[1] - η[2])) - exp(sorted_conditioner[2] * (η[1] - η[2]))) /
                (η[1] - η[2])
            B3 = (1.0 - exp(sorted_conditioner[1] * (η[1] + η[2]))) / (η[1] + η[2])

            return log(A1 * B1 + A2 * B2 + A3 * B3)
        end
        naturalparameters = [η_left, η_right]
        supp = RealInterval{Float64}(-Inf, Inf)

        return ExponentialFamilyDistribution(
            Float64,
            basemeasure,
            sufficientstatistics,
            naturalparameters,
            logpartition,
            supp
        )
    end
end

function Base.prod(::ClosedProd, left::Laplace, right::Laplace)
    location_left, scale_left = params(left)
    location_right, scale_right = params(right)

    if location_left == location_right
        return Laplace(location_left, scale_left * scale_right / (scale_left + scale_right))
    else
        ef_left = convert(KnownExponentialFamilyDistribution, left)
        ef_right = convert(KnownExponentialFamilyDistribution, right)

        (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
        (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
        basemeasure = (x) -> 1.0
        sufficientstatistics = (x) -> [abs(x - conditioner_left), abs(x - conditioner_right)]
        sorted_conditioner = sort([conditioner_left, conditioner_right])
        function logpartition(η)
            A1 = exp(η[1] * conditioner_left + η[2] * conditioner_right)
            A2 = exp(-η[1] * conditioner_left + η[2] * conditioner_right)
            A3 = exp(-η[1] * conditioner_left - η[2] * conditioner_right)
            B1 = (exp(sorted_conditioner[2] * (-η[1] - η[2])) - 1.0) / (-η[1] - η[2])
            B2 =
                (exp(sorted_conditioner[1] * (η[1] - η[2])) - exp(sorted_conditioner[2] * (η[1] - η[2]))) /
                (η[1] - η[2])
            B3 = (1.0 - exp(sorted_conditioner[1] * (η[1] + η[2]))) / (η[1] + η[2])

            return log(A1 * B1 + A2 * B2 + A3 * B3)
        end
        naturalparameters = [η_left, η_right]
        supp = RealInterval{Float64}(-Inf,Inf)

        return ExponentialFamilyDistribution(
            Float64,
            basemeasure,
            sufficientstatistics,
            naturalparameters,
            logpartition,
            supp
        )
    end
end

support(::Union{<:KnownExponentialFamilyDistribution{Laplace}, <:Laplace}) = RealInterval{Float64}(-Inf,Inf)

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Laplace)
    μ, θ = params(dist)
    return KnownExponentialFamilyDistribution(Laplace, -inv(θ), μ)
end

function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{Laplace})
    return Laplace(getconditioner(exponentialfamily), -inv(getnaturalparameters(exponentialfamily)))
end

check_valid_natural(::Type{<:Laplace}, params) = length(params) == 1

check_valid_conditioner(::Type{<:Laplace}, conditioner) = true

isproper(exponentialfamily::KnownExponentialFamilyDistribution{Laplace}) =
    getnaturalparameters(exponentialfamily) < 0

logpartition(exponentialfamily::KnownExponentialFamilyDistribution{Laplace}) =
    log(-2 / getnaturalparameters(exponentialfamily))
basemeasure(::KnownExponentialFamilyDistribution{Laplace}, x) =
    one(typeof(x))

basemeasure(::Laplace, x) = one(typeof(x))

fisherinformation(ef::KnownExponentialFamilyDistribution{Laplace}) = 1 / getnaturalparameters(ef)^2

function fisherinformation(dist::Laplace)
    # Obtained by using the weak derivative of the logpdf with respect to location parameter. Which results in sign function.
    # Expectation of sign function will be zero and expectation of square of sign will be 1. 
    b = scale(dist)
    return [1/b^2 0; 0 1/b^2]
end

function sufficientstatistics(ef::KnownExponentialFamilyDistribution{Laplace}, x)
    μ = getconditioner(ef)
    return abs(x - μ)
end

function sufficientstatistics(dist::Laplace, x)
    μ, _ = params(dist)
    return abs(x - μ)
end
