export MvNormalWishart
using Distributions
import StatsFuns: logmvgamma
using Random
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
        # @assert κ > 0 "κ parameter should be positive. Please fix κ = $(κ) "
        # @assert ν > length(μ) "ν parameter should be greater than the dimension. Please fix ν = $(ν) "
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
dim(ef::KnownExponentialFamilyDistribution{MvNormalWishart}) = length(getindex(getnaturalparameters(ef), 1))

prod_closed_rule(::Type{<:MvNormalWishart}, ::Type{<:MvNormalWishart}) = ClosedProd()

check_valid_natural(::Type{<:MvNormalWishart}, params) = length(params) === 4

function Distributions.pdf(dist::MvNormalWishart, x)
    ef = convert(KnownExponentialFamilyDistribution, dist)
    η = getnaturalparameters(ef)
    suffs = sufficientstatistics(ef)(first(x), getindex(x, 2))
    return basemeasure(dist, x) * exp(mapreduce(d -> tr(η[d]' * suffs[d]), +, 1:4) - logpartition(ef))
end

Distributions.logpdf(dist::MvNormalWishart, x) = log(pdf(dist, x))

sufficientstatistics(::Union{<:KnownExponentialFamilyDistribution{MvNormalWishart}, <:MvNormalWishart}) =
    (x, Λ) -> [Λ * x, Λ, x' * Λ * x, chollogdet(Λ)]

sufficientstatistics(union::Union{<:KnownExponentialFamilyDistribution{MvNormalWishart}, <:MvNormalWishart}, x) = sufficientstatistics(union)(x...)
function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::MvNormalWishart)
    μ, Ψ, κ, ν = params(dist)
    η1 = κ * μ
    η2 = (-1 / 2) * (inv(Ψ) + κ * μ * μ')
    η3 = -κ / 2
    η4 = (ν - dim(dist)) / 2
    η = [η1, η2, η3, η4]

    return KnownExponentialFamilyDistribution(MvNormalWishart, η)
end
function Base.convert(::Type{Distribution}, exponentialfamily::KnownExponentialFamilyDistribution{MvNormalWishart})
    d = dim(exponentialfamily)
    η = getnaturalparameters(exponentialfamily)
    η1 = first(η)
    η2 = getindex(η, 2)
    η3 = getindex(η, 3)
    η4 = getindex(η, 4)
    return MvNormalWishart(-η1 / (2 * η3), inv(-2 * η2 + η1 * η1' / (2 * η3)), -2 * η3, d + 2 * η4)
end

function logpartition(exponentialfamily::KnownExponentialFamilyDistribution{MvNormalWishart})
    d = dim(exponentialfamily)
    η = getnaturalparameters(exponentialfamily)
    η1 = first(η)
    η2 = getindex(η, 2)
    η3 = getindex(η, 3)
    η4 = getindex(η, 4)

    term1 = -(d / 2) * log(-2 * (η3))
    term2 = -((d + 2 * η4) / 2) * logdet(-2 * η2 + η1 * η1' / (2 * η3))
    term3 = log(2) * d * (d + 2 * η4) / 2
    term4 = logmvgamma(d, (d + 2 * η4) / 2)

    return (term1 + term2 + term3 + term4)
end

function isproper(exponentialfamily::KnownExponentialFamilyDistribution{MvNormalWishart})
    η = getnaturalparameters(exponentialfamily)
    η2 = getindex(η, 2)
    η3 = getindex(η, 3)
    η4 = getindex(η, 4)
    return isposdef(-η2) && η3 > 0 && η4 < 0
end

basemeasure(d::Union{<:KnownExponentialFamilyDistribution{MvNormalWishart}, <:MvNormalWishart}, x) =
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
