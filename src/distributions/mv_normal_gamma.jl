export MvNormalGamma
using Distributions
import StatsFuns: loggamma
using Random
using LinearAlgebra
using DomainSets

"""
    MvNormalGamma{T, M <: AbstractVector{T}, L <: AbstractMatrix{T}, A <: Real, B <: Real} <: ContinuousMultivariateDistribution

A multivariate normal-gamma distribution. This is the joint distribution of a multivariate
normal random vector `x` with mean `μ` and precision `τΛ`, and a gamma-distributed scalar
precision `τ` with shape `α` and rate `β`:

```
p(x, τ) = N(x | μ, (τΛ)⁻¹) ⋅ Gamma(τ | α, β)
```

It is the multivariate generalization of [`NormalGamma`](@ref): the scalar precision-scaling
`λ` becomes a `d × d` positive-definite precision-structure matrix `Λ`. Setting `d = 1`
recovers `NormalGamma` exactly.

# Fields
- `μ::M`: The mean vector of the multivariate normal distribution.
- `Λ::L`: The precision-structure matrix of the multivariate normal distribution (positive-definite).
- `α::A`: The shape parameter of the gamma distribution.
- `β::B`: The rate parameter of the gamma distribution.
"""
struct MvNormalGamma{T, M <: AbstractVector{T}, L <: AbstractMatrix{T}, A <: Real, B <: Real} <:
       ContinuousMultivariateDistribution
    μ::M
    Λ::L
    α::A
    β::B

    function MvNormalGamma(
        μ::M,
        Λ::L,
        α::A,
        β::B
    ) where {T, M <: AbstractVector{T}, L <: AbstractMatrix{T}, A <: Real, B <: Real}
        new{T, M, L, A, B}(μ, Λ, α, β)
    end

    function MvNormalGamma(
        μ::M,
        Λ::L,
        α::A,
        β::B
    ) where {T1, T2, M <: AbstractVector{T1}, L <: AbstractMatrix{T2}, A <: Real, B <: Real}
        T = promote_type(T1, T2)
        μ_new = convert(AbstractVector{T}, μ)
        Λ_new = convert(AbstractMatrix{T}, Λ)
        return new{T, typeof(μ_new), typeof(Λ_new), A, B}(μ_new, Λ_new, α, β)
    end
end

BayesBase.params(d::MvNormalGamma)   = (d.μ, d.Λ, d.α, d.β)
BayesBase.location(d::MvNormalGamma) = first(params(d))
BayesBase.scale(d::MvNormalGamma)    = getindex(params(d), 2)
BayesBase.shape(d::MvNormalGamma)    = getindex(params(d), 3)
BayesBase.rate(d::MvNormalGamma)     = getindex(params(d), 4)

# Joint mean: (E[x], E[τ]) = (μ, α/β)
BayesBase.mean(d::MvNormalGamma) = (d.μ, d.α / d.β)

Base.eltype(::MvNormalGamma{T}) where {T} = T
Base.length(d::MvNormalGamma) = length(d.μ) + 1
Base.size(d::MvNormalGamma) = (length(d),)

function BayesBase.var(d::MvNormalGamma)
    d.α > one(d.α) || error("`var` of `MvNormalGamma` is not defined for `α < 1`")
    # Marginal covariance of x is (β / (α-1)) Λ⁻¹ ; variance of τ is α/β²
    return (d.β / (d.α - one(d.α)) * cholinv(d.Λ), d.α / (d.β^2))
end

function BayesBase.cov(d::MvNormalGamma)
    d.α > one(d.α) || error("`cov` of `MvNormalGamma` is not defined for `α < 1`")
    return var(d)
end

function BayesBase.std(d::MvNormalGamma)
    d.α > one(d.α) || error("`std` of `MvNormalGamma` is not defined for `α < 1`")
    cx, vτ = var(d)
    # Per-component standard deviations of `x` (sqrt of the marginal variances) and of `τ`.
    return (sqrt.(diag(cx)), sqrt(vτ))
end

# Differential entropy H = −E[log p]. The base measure is constant, so this follows from the
# joint moments: E[log τ] = ψ(α)−log β, E[τ] = α/β, and E[τ (x−μ)ᵀΛ(x−μ)] = tr(I_d) = d.
function BayesBase.entropy(dist::MvNormalGamma)
    (μ, Λ, α, β) = params(dist)
    d = length(μ)
    return loggamma(α) - α * log(β) - logdet(Λ) / 2 + (d / 2) * (one(α) + log(twoπ)) + α -
           (α + d / 2 - one(α)) * (SpecialFunctions.digamma(α) - log(β))
end

# A sample is packed as a vector `[x₁, …, x_d, τ]`: the first `d` entries are the mean
# components and the last entry is the precision `τ`.
function BayesBase.logpdf(dist::MvNormalGamma, x::AbstractVector{<:Real})
    (μ, Λ, α, β) = params(dist)
    d = length(μ)
    xμ = view(x, 1:d)
    τ = x[d+1]
    diff = xμ - μ

    constants = α * log(β) + (1 / 2) * logdet(Λ) - loggamma(α) - (d / 2) * log(twoπ)
    term1 = (α + d / 2 - 1) * log(τ) - β * τ
    term2 = -τ * dot(diff, Λ, diff) / 2

    return constants + term1 + term2
end

BayesBase.pdf(dist::MvNormalGamma, x::AbstractVector{<:Real}) = exp(logpdf(dist, x))

function BayesBase.rand!(rng::AbstractRNG, dist::MvNormalGamma, container::AbstractVector)
    (μ, Λ, α, β) = params(dist)
    d = length(μ)
    τ = rand(rng, GammaShapeRate(α, β))
    container[d+1] = τ
    container[1:d] = rand(rng, MvNormalMeanPrecision(μ, τ * Λ))
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::MvNormalGamma, container::AbstractVector{T}) where {T <: Vector}
    for i in eachindex(container)
        rand!(rng, dist, container[i])
    end
    return container
end

function BayesBase.rand(rng::AbstractRNG, dist::MvNormalGamma)
    container = Vector{Float64}(undef, length(dist.μ) + 1)
    rand!(rng, dist, container)
    return container
end

function BayesBase.rand(rng::AbstractRNG, dist::MvNormalGamma, nsamples::Int)
    container = Vector{Vector{Float64}}(undef, nsamples)
    d = length(dist.μ)
    for i in eachindex(container)
        container[i] = Vector{Float64}(undef, d + 1)
        rand!(rng, dist, container[i])
    end
    return container
end

BayesBase.default_prod_rule(::Type{<:MvNormalGamma}, ::Type{<:MvNormalGamma}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalGamma, right::MvNormalGamma)
    (μleft, Λleft, αleft, βleft) = params(left)
    (μright, Λright, αright, βright) = params(right)
    d = length(μleft)

    Λ = Λleft + Λright
    μ = cholinv(Λ) * (Λleft * μleft + Λright * μright)
    # Natural parameters add under multiplication; mapping back gives α = αˡ + αʳ + d/2 − 1
    # (reduces to the scalar `NormalGamma` rule αˡ + αʳ − 1/2 when d = 1).
    α = αleft + αright + d / 2 - 1
    β =
        βleft + βright + dot(μleft, Λleft, μleft) / 2 + dot(μright, Λright, μright) / 2 -
        dot(μ, Λ, μ) / 2

    return MvNormalGamma(μ, Λ, α, β)
end

struct MvNormalGammaDomain <: Domain{AbstractVector} end

Base.eltype(::MvNormalGammaDomain) = AbstractVector
Base.in(v, ::MvNormalGammaDomain) = length(v) >= 2 && all(isreal, v) && v[end] > 0

BayesBase.support(::Type{MvNormalGamma}) = MvNormalGammaDomain()

# Natural parametrization
function isproper(::NaturalParametersSpace, ::Type{MvNormalGamma}, η, conditioner)
    isnothing(conditioner) || return false
    (any(isnan, η) || any(isinf, η)) && return false
    len = length(η)
    d = _mvng_dim(len)
    (d < 1 || d^2 + d + 2 != len) && return false

    (η1, η2, η3, η4) = unpack_parameters(MvNormalGamma, η)
    Λ = -2 * η2
    isposdef(Λ) || return false
    α = η3 - d / 2 + 1
    β = -η4 - dot(η1, cholinv(Λ), η1) / 2
    return α > 0 && β > 0
end

function isproper(::DefaultParametersSpace, ::Type{MvNormalGamma}, θ, conditioner)
    isnothing(conditioner) || return false
    (any(isnan, θ) || any(isinf, θ)) && return false
    len = length(θ)
    d = _mvng_dim(len)
    (d < 1 || d^2 + d + 2 != len) && return false

    (_, Λ, α, β) = unpack_parameters(MvNormalGamma, θ)
    return isposdef(Λ) && α > 0 && β > 0
end

function (::MeanToNatural{MvNormalGamma})(tuple_of_θ::Tuple{Any, Any, Any, Any})
    (μ, Λ, α, β) = tuple_of_θ
    d = length(μ)
    η1 = Λ * μ
    η2 = -Λ / 2
    η3 = α + d / 2 - 1
    η4 = -β - dot(μ, Λ, μ) / 2
    return (η1, η2, η3, η4)
end

function (::NaturalToMean{MvNormalGamma})(tuple_of_η::Tuple{Any, Any, Any, Any})
    (η1, η2, η3, η4) = tuple_of_η
    d = length(η1)
    Λ = -2 * η2
    Λinv = cholinv(Λ)
    μ = Λinv * η1
    α = η3 - d / 2 + 1
    β = -η4 - dot(η1, Λinv, η1) / 2
    return (μ, Λ, α, β)
end

# Recover the location dimension `d` from a packed parameter vector of length `d² + d + 2`.
_mvng_dim(len) = Int64((-1 + isqrt(1 - 4 * (2 - len))) / 2)

function unpack_parameters(::Type{MvNormalGamma}, packed)
    len = length(packed)
    d = _mvng_dim(len)

    @inbounds η1 = view(packed, 1:d)
    @inbounds η2 = reshape(view(packed, (d+1):(d^2+d)), d, d)
    @inbounds η3 = packed[d^2+d+1]
    @inbounds η4 = packed[d^2+d+2]

    return (η1, η2, η3, η4)
end

isbasemeasureconstant(::Type{MvNormalGamma}) = ConstantBaseMeasure()

# Base measure (2π)^(-d/2), where d = length(x) - 1.
getbasemeasure(::Type{MvNormalGamma}) = (x) -> (twoπ)^(-(length(x) - 1) / 2)

# x is a (d+1)-vector: first d entries are the mean, last entry is the precision τ.
getsufficientstatistics(::Type{MvNormalGamma}) = (
    x -> x[end] * view(x, 1:(length(x)-1)),
    x -> x[end] * (view(x, 1:(length(x)-1)) * view(x, 1:(length(x)-1))'),
    x -> log(x[end]),
    x -> x[end]
)

getlogpartition(::NaturalParametersSpace, ::Type{MvNormalGamma}) = (η) -> begin
    (η1, η2, η3, η4) = unpack_parameters(MvNormalGamma, η)
    d = length(η1)
    Λ = -2 * η2
    α = η3 - d / 2 + 1
    β = -η4 - dot(η1, cholinv(Λ), η1) / 2
    return loggamma(α) - α * log(β) - (1 / 2) * logdet(Λ)
end

getgradlogpartition(::NaturalParametersSpace, ::Type{MvNormalGamma}) = (η) -> begin
    (η1, η2, η3, η4) = unpack_parameters(MvNormalGamma, η)
    d = length(η1)
    Λ = -2 * η2
    Λinv = cholinv(Λ)
    μ = Λinv * η1
    α = η3 - d / 2 + 1
    β = -η4 - dot(η1, Λinv, η1) / 2

    # The gradient of the log-partition equals E[T(x,τ)]:
    #   E[τx]    = μ ⋅ α/β
    #   E[τxxᵀ]  = Λ⁻¹ + (α/β) μμᵀ
    #   E[log τ] = digamma(α) - log β
    #   E[τ]     = α/β
    dη1 = (α / β) * μ
    dη2 = Λinv + (α / β) * (μ * μ')
    dη3 = SpecialFunctions.digamma(α) - log(β)
    dη4 = α / β

    return vcat(dη1, vec(dη2), dη3, dη4)
end

# Fisher information in the natural parameter space.
#
# Because `η₂` is stored as a full `d × d` matrix while the family only depends on its
# symmetric part, the literal Hessian of the log-partition w.r.t. the free entries does not
# match `ForwardDiff` (which also sees the antisymmetric directions), and the textbook
# `Cov(T)` is rank-deficient (the sufficient statistic `τxxᵀ` is symmetric). The same issue
# applies to `MvNormalWishart`, which is why the Fisher-against-Hessian/Jacobian checks are
# disabled there too. We therefore build a full-rank, symmetric, positive-definite Fisher via
# the exact metric transform `Fη = Jᵀ Fθ J`, where `J = ∂θ/∂η` is the Jacobian of
# `NaturalToMean` and `Fθ` is the mean-space Fisher below. This reduces exactly to the scalar
# `NormalGamma` Fisher when `d = 1`.
getfisherinformation(::NaturalParametersSpace, ::Type{MvNormalGamma}) = (η) -> begin
    (η1, η2, η3, η4) = unpack_parameters(MvNormalGamma, η)
    d = length(η1)
    Λ = -2 * η2
    W = cholinv(Λ)
    μ = W * η1
    α = η3 - d / 2 + 1
    β = -η4 - dot(η1, W, η1) / 2

    J = _mvng_natural_to_mean_jacobian(μ, W, d, eltype(η))
    Fθ = _mvng_fisher_mean(Λ, W, α, β, d, eltype(η))
    return J' * Fθ * J
end

getfisherinformation(::DefaultParametersSpace, ::Type{MvNormalGamma}) = (θ) -> begin
    (μ, Λ, α, β) = unpack_parameters(MvNormalGamma, θ)
    d = length(μ)
    W = cholinv(Λ)
    return _mvng_fisher_mean(Λ, W, α, β, d, eltype(θ))
end

# Mean-space Fisher information. The block for `μ` decouples from the rest (third central
# Gaussian moments vanish); the `Λ` block uses the full-rank `½(Λ⁻¹ ⊗ Λ⁻¹)` convention; the
# `(α, β)` block is the standard gamma Fisher information.
function _mvng_fisher_mean(Λ, W, α, β, d, T)
    n = d^2 + d + 2
    F = zeros(T, n, n)
    r1 = 1:d
    r2 = (d+1):(d^2+d)
    i3 = d^2 + d + 1
    i4 = d^2 + d + 2
    @inbounds begin
        F[r1, r1] = (α / β) * Λ
        F[r2, r2] = (1 / 2) * kron(W, W)
        F[i3, i3] = SpecialFunctions.trigamma(α)
        F[i4, i4] = α / β^2
        F[i3, i4] = -1 / β
        F[i4, i3] = -1 / β
    end
    return F
end

# Jacobian J = ∂θ/∂η of `NaturalToMean`, with θ = (μ, vec(Λ), α, β) and η = (η₁, vec(η₂), η₃, η₄).
function _mvng_natural_to_mean_jacobian(μ, W, d, T)
    n = d^2 + d + 2
    J = zeros(T, n, n)
    r1 = 1:d
    r2 = (d+1):(d^2+d)
    i3 = d^2 + d + 1
    i4 = d^2 + d + 2
    @inbounds begin
        # μ = Λ⁻¹ η₁,  Λ = -2η₂
        J[r1, r1] = W
        J[r1, r2] = 2 * kron(μ', W)
        # vec(Λ) = -2 vec(η₂)
        J[r2, r2] = Matrix{T}(-2 * I, d^2, d^2)
        # α = η₃ - d/2 + 1
        J[i3, i3] = one(T)
        # β = -η₄ - ½ η₁ᵀ Λ⁻¹ η₁
        J[i4, r1] = -μ'
        J[i4, r2] = -vec(μ * μ')'
        J[i4, i4] = -one(T)
    end
    return J
end

# Mean parametrization

getlogpartition(::DefaultParametersSpace, ::Type{MvNormalGamma}) = (θ) -> begin
    (_, Λ, α, β) = unpack_parameters(MvNormalGamma, θ)
    return loggamma(α) - α * log(β) - (1 / 2) * logdet(Λ)
end
