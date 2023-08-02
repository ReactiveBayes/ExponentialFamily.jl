export MvNormalWishart
using Distributions
import StatsFuns: logmvgamma
using Random

"""
    MvNormalWishart{T, M <: AbstractArray{T}, V <: AbstractMatrix{T}, K <: Real, N <: Real} <: ContinuousMatrixDistribution

    A multivariate normal-Wishart distribution, where `T` is the element type of the arrays `M` and matrices `V`, and `K` and `N` are real numbers. This distribution is a joint distribution of a multivariate normal random variable with mean `μ` and a Wishart-distributed random matrix with scale matrix `Ψ`, degrees of freedom `ν`, and the scalar `κ` as a scaling parameter.

    # Fields
    - `μ::M`: The mean vector of the multivariate normal distribution.
    - `Ψ::V`: The scale matrix of the Wishart distribution.
    - `κ::K`: The scaling parameter of the Wishart distribution.
    - `ν::N`: The degrees of freedom of the Wishart distribution
"""
struct MvNormalWishart{T, M <: AbstractArray{T}, V <: AbstractMatrix{T}, K <: Real, N <: Real} <:
       ContinuousMatrixDistribution
    μ::M
    Ψ::V
    κ::K
    ν::N

    function MvNormalWishart(
        μ::M,
        Ψ::V,
        κ::K,
        ν::N
    ) where {T, M <: AbstractArray{T}, V <: AbstractMatrix{T}, K <: Real, N <: Real}
        new{T, M, V, K, N}(μ, Ψ, κ, ν)
    end
end

params(d::MvNormalWishart) = (d.μ, d.Ψ, d.κ, d.ν)
location(d::MvNormalWishart) = first(params(d))
scatter(d::MvNormalWishart) = getindex(params(d), 2)
invscatter(d::MvNormalWishart) = cholinv(getindex(params(d), 2))
scale(d::MvNormalWishart) = getindex(params(d), 3)
dof(d::MvNormalWishart) = getindex(params(d), 4)
dim(d::MvNormalWishart) = length(d.μ)
function dim(ef::ExponentialFamilyDistribution{MvNormalWishart})
    len = length(getnaturalparameters(ef))

    return Int64((-1 + isqrt(1 - 4 * (2 - len))) / 2)
end

closed_prod_rule(::Type{<:MvNormalWishart}, ::Type{<:MvNormalWishart}) = ClosedProd()

function check_valid_natural(::Type{<:MvNormalWishart}, params)
    return length(params) >= 4 && length(params) % 2 == 0
end

function Distributions.pdf(dist::MvNormalWishart, x)
    ef = convert(ExponentialFamilyDistribution, dist)
    η = getnaturalparameters(ef)
    suffs = sufficientstatistics(ef)(first(x), getindex(x, 2))
    return basemeasure(dist, x) * exp(dot(η, suffs) - logpartition(ef))
end

Distributions.logpdf(dist::MvNormalWishart, x) = log(pdf(dist, x))

sufficientstatistics(::Union{<:ExponentialFamilyDistribution{MvNormalWishart}, <:MvNormalWishart}) =
    (x, Λ) -> vcat(Λ * x, vec(Λ), dot(x, Λ, x), logdet(Λ))

sufficientstatistics(union::Union{<:ExponentialFamilyDistribution{MvNormalWishart}, <:MvNormalWishart}, x) =
    sufficientstatistics(union)(x[1], x[2])

function pack_naturalparameters(dist::MvNormalWishart)
    μ, Ψ, κ, ν = params(dist)
    η1 = κ * μ
    η2 = (-1 / 2) * (vec(cholinv(Ψ)) + κ * kron(μ, μ))
    η3 = -κ / 2
    η4 = (ν - dim(dist)) / 2
    return vcat(η1, η2, η3, η4)
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:MvNormalWishart})
    η = getnaturalparameters(ef)
    d = dim(ef)

    @inbounds η1 = view(η, 1:d)
    @inbounds η2 = reshape(view(η, d+1:d^2+d), d, d)
    @inbounds η3 = η[d^2+d+1]
    @inbounds η4 = η[d^2+d+2]

    return η1, η2, η3, η4
end

Base.convert(::Type{ExponentialFamilyDistribution}, dist::MvNormalWishart) =
    ExponentialFamilyDistribution(MvNormalWishart, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{MvNormalWishart})
    d = dim(exponentialfamily)
    η1, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)
    return MvNormalWishart(-HALF * η1 / η3, cholinv(-2 * η2 + HALF * η1 * η1' / (η3)), -2 * η3, d + 2 * η4)
end

function logpartition(exponentialfamily::ExponentialFamilyDistribution{MvNormalWishart})
    d = dim(exponentialfamily)
    η1, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)

    term1 = -(d * HALF) * log(-2 * (η3))
    term2 = -((d + 2 * η4) * HALF) * logdet(-2 * η2 + HALF * η1 * η1' / (η3))
    term3 = LOG2 * d * (d + 2 * η4) * HALF
    term4 = logmvgamma(d, (d + 2 * η4) * HALF)

    return (term1 + term2 + term3 + term4)
end

function isproper(exponentialfamily::ExponentialFamilyDistribution{MvNormalWishart})
    _, η2, η3, η4 = unpack_naturalparameters(exponentialfamily)
    return isposdef(-η2) && η3 > 0 && η4 < 0
end

basemeasure(d::Union{<:ExponentialFamilyDistribution{MvNormalWishart}, <:MvNormalWishart}) =
    1 / (2 * pi)^(dim(d) / 2)
basemeasure(d::Union{<:ExponentialFamilyDistribution{MvNormalWishart}, <:MvNormalWishart}, x) =
    1 / (2 * pi)^(dim(d) / 2)

function Random.rand!(rng::AbstractRNG, dist::MvNormalWishart, container::Tuple{AbstractVector, AbstractMatrix})
    μ, Ψ, κ, ν = params(dist)
    rand!(rng, Wishart(ν, Ψ), container[2])
    rand!(rng, MvNormalMeanPrecision(μ, κ * container[2]), container[1])
    return container
end

function Random.rand!(rng::AbstractRNG, dist::MvNormalWishart, container::AbstractVector{T}) where {T <: Tuple}
    for i in eachindex(container)
        rand!(rng, dist, container[i])
    end
    return container
end

function Random.rand(rng::AbstractRNG, dist::MvNormalWishart{T}) where {T}
    container = (Vector{T}(undef, dim(dist)), Matrix{T}(undef, (dim(dist), dim(dist))))
    return rand!(rng, dist, container)
end

function Random.rand(rng::AbstractRNG, dist::MvNormalWishart{T}, nsamples::Int64) where {T}
    container = Vector{Tuple{Vector{T}, Matrix{T}}}(undef, nsamples)
    for i in eachindex(container)
        container[i] = (Vector{T}(undef, dim(dist)), Matrix{T}(undef, (dim(dist), dim(dist))))
        rand!(rng, dist, container[i])
    end
    return container
end
