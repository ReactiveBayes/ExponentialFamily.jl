export VonMisesFisher
using Distributions
import Distributions: VonMisesFisher
import SpecialFunctions: besseli, gamma
import LinearAlgebra: norm
import FillArrays: Eye
import HCubature: hquadrature
using HypergeometricFunctions

BayesBase.vague(::Type{<:VonMisesFisher}, dims::Int64) = VonMisesFisher(zeros(dims), tiny)

BayesBase.default_prod_rule(::Type{<:VonMisesFisher}, ::Type{<:VonMisesFisher}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::VonMisesFisher, right::VonMisesFisher)
    (μleft, κleft) = params(left)
    (μright, κright) = params(right)
    weightedsum = μleft * κleft + μright * κright
    κ = norm(weightedsum)
    μ = weightedsum / κ
    return VonMisesFisher(μ, κ)
end

function BayesBase.mean(dist::VonMisesFisher)
    (μ, κ) = Distributions.params(dist)
    p = length(μ)
    factor = besseli(0.5p, κ) / besseli(0.5p - 1, κ)
    return factor * μ
end

function BayesBase.cov(dist::VonMisesFisher)
    (μ, κ) = Distributions.params(dist)
    ν = length(μ)
    rb = besseli(ν / 2, κ) * inv(besseli((ν / 2) - 1, κ))
    return (rb * inv(κ)) * Eye{Float64}(ν) + (besseli((ν / 2) + 1, κ) / besseli((ν / 2) - 1, κ) - rb^2) * μ * μ'
end

BayesBase.var(dist::VonMisesFisher) = diag(cov(dist))
BayesBase.std(dist::VonMisesFisher) = sqrt.(var(dist))

function BayesBase.insupport(ef::ExponentialFamilyDistribution{VonMisesFisher}, x)
    return length(getnaturalparameters(ef)) == length(x) && Distributions.isunitvec(x)
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{VonMisesFisher}, η, conditioner) = isnothing(conditioner) && length(η) > 1 && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{VonMisesFisher}, θ, conditioner) =
    isnothing(conditioner) && length(θ) > 1 && getindex(θ, 2) >= 0 && norm(first(θ)) ≈ 1.0 && all(!isinf, θ) && all(!isnan, θ)

function (::MeanToNatural{VonMisesFisher})(tuple_of_θ::Tuple{Any, Any})
    (μ, κ) = tuple_of_θ
    return (κ * μ,)
end

function (::NaturalToMean{VonMisesFisher})(tuple_of_η::Tuple{Any})
    (η,) = tuple_of_η
    κ = norm(η)
    μ = η / κ
    return (μ, κ)
end

function unpack_parameters(::MeanParametersSpace, ::Type{VonMisesFisher}, packed)
    (μ, κ) = (view(packed, 1:length(packed)-1), packed[end])

    return (μ, κ)
end

function unpack_parameters(::NaturalParametersSpace, ::Type{VonMisesFisher}, packed)
    return (packed,)
end

isbasemeasureconstant(::Type{VonMisesFisher}) = ConstantBaseMeasure()

getbasemeasure(::Type{VonMisesFisher}) = (x) -> (inv2π)^(length(x) / 2)
getsufficientstatistics(::Type{VonMisesFisher}) = (identity,)

getlogpartition(::NaturalParametersSpace, ::Type{VonMisesFisher}) = (η) -> begin
    κ = sqrt(η' * η)
    p = length(η)
    return log(besseli((p / 2) - 1, κ)) - ((p / 2) - 1) * log(κ)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{VonMisesFisher}) = (η) -> begin
    κ = sqrt(dot(η, η))
    p = length(η)
    term1 = - ((p / 2) - 1) / κ
    term2 = ((p / 2) - 1)/κ  +  besseli((p / 2), κ)/besseli((p / 2) - 1, κ)
    term3 = (term1 + term2)/(κ)
    return term3*η
end

getfisherinformation(::NaturalParametersSpace, ::Type{VonMisesFisher}) = (η) -> begin
    u = norm(η)
    p = length(η)

    bessel3 = besseli(p / 2 - 3, u)
    bessel2 = besseli(p / 2 - 2, u)
    bessel1 = besseli(p / 2 - 1, u)
    bessel0 = besseli(p / 2, u)
    bessel4 = besseli(p / 2 + 1, u)

    f1 = (1 / 2) * (bessel0 + bessel2)
    f2 = inv(bessel1)
    f3 = (p / 2 - 1) / u
    f4 = η / u

    delu = η' / u
    df1  = (1 / 4) * (bessel4 + 2 * bessel1 + bessel3) * delu
    df2  = ((-1 / 2) * (bessel2 + bessel0) / bessel1^2) * delu
    df3  = (-(p / 2 - 1) / u^2) * delu
    df4  = Eye(p) / u - η * η' / u^3

    return f4 * df1 * f2 + f4 * f1 * df2 + f1 * f2 * df4 - f4 * df3 - f3 * df4
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{VonMisesFisher}) = (θ) -> begin
    (μ, κ) = unpack_parameters(MeanParametersSpace(), VonMisesFisher, θ)
    p = length(μ)
    return log(besseli((p / 2) - 1, κ)) - ((p / 2) - 1) * log(κ)
end

getfisherinformation(::MeanParametersSpace, ::Type{VonMisesFisher}) =
    (θ) -> begin
        (μ, k) = unpack_parameters(MeanParametersSpace(), VonMisesFisher, θ)
        p = length(μ)

        bessel3 = besseli((p / 2) - 3, k)
        bessel2 = besseli((p / 2) - 2, k)
        bessel1 = besseli((p / 2) - 1, k)
        bessel0 = besseli((p / 2), k)
        bessel4 = besseli((p / 2) + 1, k)

        tmp =
            (p / 2 - 1) / k^2 + (1 / 4) * (bessel3 + 2 * bessel1 + bessel4) / bessel1 -
            (1 / 4) * (bessel2 + bessel0)^2 / bessel1^2
        Ap = bessel0 / bessel1
        tmp2 = (1 - Ap * p / k - Ap^2) * μ * μ' + inv(k) * Ap * Eye(p)
        return [k^2*tmp2 -Ap*μ; -Ap*μ' tmp]
    end
