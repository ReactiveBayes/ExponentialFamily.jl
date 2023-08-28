export VonMisesFisher
using Distributions
import Distributions: VonMisesFisher
import SpecialFunctions: besseli
import LinearAlgebra: norm

vague(::Type{<:VonMisesFisher}, dims::Int64) = VonMisesFisher(zeros(dims), tiny)

function Distributions.mean(dist::VonMisesFisher)
    (μ, κ) = Distributions.params(dist)

    p = length(μ)
    factor = besseli(0.5p, κ) / besseli(0.5p - 1, κ)
    return factor * μ
end

#### Write these functions
function Distributions.cov(dist::VonMisesFisher)
    return 1.0
end

Distributions.var(dist::VonMisesFisher) = 1.0
Distributions.std(dist::VonMisesFisher) = 1.0

function insupport(ef::ExponentialFamilyDistribution{VonMisesFisher}, x::Vector) 
    return length(getnaturalparameters(ef)) == length(x) && Distributions.isunitvec(x)
end

# Natural parametrization

isproper(::NaturalParametersSpace, ::Type{VonMisesFisher}, η, conditioner) = isnothing(conditioner) && length(η) > 1 && all(!isinf, η) && all(!isnan, η)
isproper(::MeanParametersSpace, ::Type{VonMisesFisher}, θ, conditioner) = isnothing(conditioner) && length(θ) > 1 && getindex(θ,2) >= 0  && norm(first(θ)) ≈ 1.0 && all(!isinf, θ) && all(!isnan, θ)

function (::MeanToNatural{VonMisesFisher})(tuple_of_θ::Tuple{Any, Any})
    (μ, κ) = tuple_of_θ
    return (κ*μ , )
end

function (::NaturalToMean{VonMisesFisher})(tuple_of_η::Tuple{Any})
    (η, ) = tuple_of_η
    κ = norm(η)
    μ = η / κ
    return (μ, κ)
end

function unpack_parameters(::MeanParametersSpace, ::Type{VonMisesFisher}, packed)
    (μ, κ) = (view(packed, 1:length(packed)-1), packed[end])

    return (μ, κ)
end

function unpack_parameters(::NaturalParametersSpace, ::Type{VonMisesFisher}, packed)
    return (packed, )
end

isbasemeasureconstant(::Type{VonMisesFisher}) = ConstantBaseMeasure()

getbasemeasure(::Type{VonMisesFisher}) = (x) -> (inv2π)^(length(x)/2)
getsufficientstatistics(::Type{VonMisesFisher}) = (identity, )

getlogpartition(::NaturalParametersSpace, ::Type{VonMisesFisher}) = (η) -> begin
    κ = sqrt(η' * η)
    p = length(η)
    return log(besseli((p / 2) - 1, κ)) - ((p / 2) - 1) * log(κ)
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
    df4  = diageye(p) / u - η * η' / u^3

    return f4 * df1 * f2 + f4 * f1 * df2 + f1 * f2 * df4 - f4 * df3 - f3 * df4
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{VonMisesFisher}) = (θ) -> begin
    (μ, κ) = unpack_parameters(MeanParametersSpace(), VonMisesFisher,θ )
    p = length(μ)
    return log(besseli((p / 2) - 1, κ)) - ((p / 2) - 1) * log(κ)
end

getfisherinformation(::MeanParametersSpace, ::Type{VonMisesFisher}) = (θ) -> begin
    (μ, k) = unpack_parameters(MeanParametersSpace(), VonMisesFisher,θ )
    p = length(μ)

    bessel3 = besseli(p / 2 - 3, k)
    bessel2 = besseli(p / 2 - 2, k)
    bessel1 = besseli(p / 2 - 1, k)
    bessel0 = besseli(p / 2, k)
    bessel4 = besseli(p / 2 + 1, k)

    tmp =
    (p / 2 - 1) / k^2 + (1 / 4) * (bessel3 + 2 * bessel1 + bessel4) / bessel1 -
    (1 / 4) * (bessel2 + bessel0)^2 / bessel1^2
    Ap = bessel0 / bessel1
    tmp2 = (1 - Ap * p / k - Ap^2) * μ * μ' + inv(k) * Ap * diageye(p)
    return [k^2*tmp2 -Ap*μ; -Ap*μ' tmp]
end





# isproper(exponentialfamily::ExponentialFamilyDistribution{VonMisesFisher}) =
#     all(0 .<= (getnaturalparameters(exponentialfamily)))

# function Base.convert(::Type{ExponentialFamilyDistribution}, dist::VonMisesFisher)
#     μ, κ = params(dist)
#     ExponentialFamilyDistribution(VonMisesFisher, μ * κ)
# end

# function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{VonMisesFisher})
#     κμ = getnaturalparameters(exponentialfamily)
#     κ = sqrt(κμ' * κμ)
#     μ = κμ / κ
#     return VonMisesFisher(μ, κ)
# end

# check_valid_natural(::Type{<:VonMisesFisher}, v) = length(v) >= 2

# function logpartition(exponentialfamily::ExponentialFamilyDistribution{VonMisesFisher})
#     η = getnaturalparameters(exponentialfamily)
#     ## because ||μ|| = 1 this trick obtains κ 
#     κ = sqrt(η' * η)
#     p = length(η)
#     return log(besseli((p / 2) - 1, κ)) - ((p / 2) - 1) * log(κ)
# end

# function fisherinformation(dist::VonMisesFisher)
#     μ, k = params(dist)
#     p    = length(μ)

#     bessel3 = besseli(p / 2 - 3, k)
#     bessel2 = besseli(p / 2 - 2, k)
#     bessel1 = besseli(p / 2 - 1, k)
#     bessel0 = besseli(p / 2, k)
#     bessel4 = besseli(p / 2 + 1, k)

#     tmp =
#         (p / 2 - 1) / k^2 + (1 / 4) * (bessel3 + 2 * bessel1 + bessel4) / bessel1 -
#         (1 / 4) * (bessel2 + bessel0)^2 / bessel1^2
#     Ap = bessel0 / bessel1
#     tmp2 = (1 - Ap * p / k - Ap^2) * μ * μ' + inv(k) * Ap * diageye(p)
#     return [k^2*tmp2 -Ap*μ; -Ap*μ' tmp]
# end

# function fisherinformation(ef::ExponentialFamilyDistribution{VonMisesFisher})
#     η = getnaturalparameters(ef)
#     u = norm(η)
#     p = length(η)

#     bessel3 = besseli(p / 2 - 3, u)
#     bessel2 = besseli(p / 2 - 2, u)
#     bessel1 = besseli(p / 2 - 1, u)
#     bessel0 = besseli(p / 2, u)
#     bessel4 = besseli(p / 2 + 1, u)

#     f1 = (1 / 2) * (bessel0 + bessel2)
#     f2 = inv(bessel1)
#     f3 = (p / 2 - 1) / u
#     f4 = η / u

#     delu = η' / u
#     df1  = (1 / 4) * (bessel4 + 2 * bessel1 + bessel3) * delu
#     df2  = ((-1 / 2) * (bessel2 + bessel0) / bessel1^2) * delu
#     df3  = (-(p / 2 - 1) / u^2) * delu
#     df4  = diageye(p) / u - η * η' / u^3

#     return f4 * df1 * f2 + f4 * f1 * df2 + f1 * f2 * df4 - f4 * df3 - f3 * df4
# end

# function insupport(vmf::VonMisesFisher, x::Vector)
#     return length(vmf.μ) == length(x) && dot(x, x) ≈ 1.0
# end



# basemeasure(ef::ExponentialFamilyDistribution{VonMisesFisher}) = (1 / twoπ)^(length(getnaturalparameters(ef)) * (1/2))
# function basemeasure(::Union{<:ExponentialFamilyDistribution{VonMisesFisher}, <:VonMisesFisher}, x::Vector)
#     return (1 / 2pi)^(length(x) * (1/2))
# end

# sufficientstatistics(ef::Union{<:ExponentialFamilyDistribution{VonMisesFisher}, <:VonMisesFisher}) =
#     (x) -> sufficientstatistics(ef, x)
# function sufficientstatistics(
#     ::Union{<:ExponentialFamilyDistribution{VonMisesFisher}, <:VonMisesFisher},
#     x::Vector
# )
#     return x
# end
