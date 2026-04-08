export MatrixNormalWishart

import Base: convert
import StatsFuns: logmvgamma
using Random

"""
    MatrixNormalWishart{T, M, S, K, N, C}

A Matrix Normal-Wishart distribution — the conjugate prior for the mean matrix and row-precision
of a Matrix Normal distribution.

Models the joint distribution of a matrix `M` (n×p) and a positive-definite matrix `Λ` (n×n):
- `M | Λ ~ MatrixNormal(M₀, (κΛ)⁻¹, V)`
- `Λ ~ Wishart(ν, Ψ)`

# Fields
- `M₀`: Prior mean matrix (n×p).
- `Ψ`: Wishart scale matrix (n×n).
- `κ`: Precision scaling factor (scalar > 0).
- `ν`: Wishart degrees of freedom (scalar > n−1).
- `V`: Column covariance matrix (p×p).

# Note
`V` plays the role of a fixed hyperparameter (conditioner) in the exponential family interface.
When converting to `ExponentialFamilyDistribution`, `V` is stored as the conditioner.
"""
struct MatrixNormalWishart{T, M <: AbstractMatrix{T}, S <: AbstractMatrix{T}, K <: Real, N <: Real, C <: AbstractMatrix{T}} <:
       ContinuousMultivariateMatrixvariateDistribution
    M₀::M
    Ψ::S
    κ::K
    ν::N
    V::C

    function MatrixNormalWishart(
        M₀::M,
        Ψ::S,
        κ::K,
        ν::N,
        V::C
    ) where {T, M <: AbstractMatrix{T}, S <: AbstractMatrix{T}, K <: Real, N <: Real, C <: AbstractMatrix{T}}
        new{T, M, S, K, N, C}(M₀, Ψ, κ, ν, V)
    end

    function MatrixNormalWishart(
        M₀::M,
        Ψ::S,
        κ::K,
        ν::N,
        V::C
    ) where {T1, T2, T3, M <: AbstractMatrix{T1}, S <: AbstractMatrix{T2}, K <: Real, N <: Real, C <: AbstractMatrix{T3}}
        T = promote_type(T1, T2, T3)
        M₀_new = convert(AbstractMatrix{T}, M₀)
        Ψ_new  = convert(AbstractMatrix{T}, Ψ)
        V_new  = convert(AbstractMatrix{T}, V)
        return new{T, typeof(M₀_new), typeof(Ψ_new), K, N, typeof(V_new)}(M₀_new, Ψ_new, κ, ν, V_new)
    end
end

Base.eltype(::MatrixNormalWishart{T}) where {T} = T
Base.size(d::MatrixNormalWishart) = size(d.M₀)

BayesBase.params(d::MatrixNormalWishart)  = (d.M₀, d.Ψ, d.κ, d.ν, d.V)
BayesBase.mean(d::MatrixNormalWishart)    = (d.M₀, d.ν * d.Ψ)
BayesBase.dof(d::MatrixNormalWishart)     = d.ν
BayesBase.location(d::MatrixNormalWishart) = d.M₀

ExponentialFamily.separate_conditioner(::Type{MatrixNormalWishart}, params) =
    ((params[1], params[2], params[3], params[4]), params[5])

ExponentialFamily.join_conditioner(::Type{MatrixNormalWishart}, cparams, V) =
    (cparams..., V)

function BayesBase.pdf(dist::MatrixNormalWishart, x::Tuple)
    M, Λ = x
    M₀, Ψ, κ, ν, V = params(dist)
    return pdf(MatrixNormal(M₀, cholinv(κ * Λ), V), M) * pdf(Wishart(ν, Ψ), Λ)
end

BayesBase.logpdf(dist::MatrixNormalWishart, x::Tuple) = log(pdf(dist, x))

function BayesBase.rand(rng::AbstractRNG, dist::MatrixNormalWishart{T}) where {T}
    M₀, Ψ, κ, ν, V = params(dist)
    Λ = rand(rng, Wishart(ν, Ψ))
    M = rand(rng, MatrixNormal(M₀, cholinv(κ * Λ), V))
    return (M, Λ)
end

function BayesBase.rand(rng::AbstractRNG, dist::MatrixNormalWishart{T}, nsamples::Int) where {T}
    return [rand(rng, dist) for _ in 1:nsamples]
end

BayesBase.default_prod_rule(::Type{<:MatrixNormalWishart}, ::Type{<:MatrixNormalWishart}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MatrixNormalWishart, right::MatrixNormalWishart)
    M₀l, Ψl, κl, νl, Vl = params(left)
    M₀r, Ψr, κr, νr, Vr = params(right)

    Vi_l = cholinv(Vl)

    κ  = κl + κr
    M₀ = (κl * M₀l + κr * M₀r) / κ
    Ψ  = cholinv(
        cholinv(Ψl) + κl * M₀l * Vi_l * M₀l' +
        cholinv(Ψr) + κr * M₀r * Vi_l * M₀r' -
        κ * M₀ * Vi_l * M₀'
    )
    n, p = size(M₀l)
    ν = νl + νr + p - n - 1

    return MatrixNormalWishart(M₀, Ψ, κ, ν, Vl)
end

function BayesBase.insupport(::ExponentialFamilyDistribution{MatrixNormalWishart}, x)
    return x isa Tuple && length(x) == 2 && isposdef(x[2])
end

function isproper(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, η, V)
    if isnothing(V) || any(isnan, η) || any(isinf, η)
        return false
    end
    p = size(V, 1)
    _, _, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, V)
    return η₃ < 0 && η₄ > (p - 2) / 2
end

function isproper(::DefaultParametersSpace, ::Type{MatrixNormalWishart}, θ, V)
    if isnothing(V)
        return false
    end
    M₀, Ψ, κ, ν = θ
    if any(x -> any(isnan, x) || any(isinf, x), (M₀, Ψ)) || isnan(κ) || isinf(κ) || isnan(ν) || isinf(ν)
        return false
    end
    n = size(M₀, 1)
    return κ > 0 && ν > n - 1
end

function (::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::Tuple{Any, Any, Any, Any}, V)
    M₀, Ψ, κ, ν = tuple_of_θ
    n, p = size(M₀)
    Vi    = cholinv(V)
    M₀Vi  = M₀ * Vi
    η₁ = κ * M₀Vi
    η₂ = -1 / 2 * (cholinv(Ψ) + κ * M₀Vi * M₀')
    η₃ = -κ / 2
    η₄ = (ν + p - n - 1) / 2
    return (η₁, η₂, η₃, η₄)
end

function (t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::Tuple{Any, Any, Any, Any}, V)
    η₁, η₂, η₃, η₄ = tuple_of_η
    n, p = size(η₁)
    κ  = -2η₃
    M₀ = (-1 / (2η₃)) * η₁ * V
    ν  = 2η₄ + n + 1 - p
    Ψ  = cholinv(-2η₂ + η₁ * V * η₁' / (2η₃))
    return (M₀, Ψ, κ, ν)
end

function (t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::Tuple{Any, Any, Any, Any}, V, ::Tuple{Int, Int})
    return t(tuple_of_η, V)
end

# Infer n from p = size(V,1) and total length of η.
# Packed layout: [vec(η₁) (n×p), vec(η₂) (n×n), η₃, η₄]   length = np + n² + 2
# n satisfies: n² + np − (length(η)−2) = 0  →  n = (−p + √(p²+4L)) / 2
function unpack_parameters(::Type{MatrixNormalWishart}, η, V)
    p = size(V, 1)
    L = length(η) - 2
    n = Int(round((-p + sqrt(p^2 + 4L)) / 2))
    @inbounds η₁ = reshape(view(η, 1:(n * p)), n, p)
    @inbounds η₂ = reshape(view(η, (n * p + 1):(n * p + n^2)), n, n)
    @inbounds η₃ = η[n * p + n^2 + 1]
    @inbounds η₄ = η[n * p + n^2 + 2]
    return η₁, η₂, η₃, η₄
end

isbasemeasureconstant(::Type{MatrixNormalWishart}) = ConstantBaseMeasure()

getbasemeasure(::Type{MatrixNormalWishart})    = (_) -> 1.0
getbasemeasure(::Type{MatrixNormalWishart}, V) = (_) -> 1.0

getlogbasemeasure(::Type{MatrixNormalWishart})    = (_) -> 0.0
getlogbasemeasure(::Type{MatrixNormalWishart}, V) = (_) -> 0.0

function getsufficientstatistics(::Type{MatrixNormalWishart}, V)
    Vi = cholinv(V)
    return (
        (z) -> begin (M, Λ) = z; Λ * M end,
        (z) -> begin (_, Λ) = z; Λ end,
        (z) -> begin (M, Λ) = z; tr(Vi * M' * Λ * M) end,
        (z) -> begin (_, Λ) = z; logdet(Λ) end
    )
end

getlogpartition(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, V) = (η) -> begin
    η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, V)
    n, p = size(η₁)
    ν  = 2η₄ + n + 1 - p
    Ψ  = cholinv(-2η₂ + η₁ * V * η₁' / (2η₃))
    term1 = -(n * p / 2) * log(-2η₃)
    term2 = (ν / 2) * logdet(Ψ)
    term3 = (ν * n / 2) * log(2.0)
    term4 = logmvgamma(n, ν / 2)
    term5 = (n / 2) * logdet(V)
    return term1 + term2 + term3 + term4 + term5 + (n * p / 2) * log2π
end

getlogpartition(::DefaultParametersSpace, ::Type{MatrixNormalWishart}, V) = (θ) -> begin
    M₀, Ψ, κ, ν = θ
    n, p = size(M₀)
    term1 = (n * p / 2) * log2π
    term2 = (n / 2) * logdet(V)
    term3 = -(n * p / 2) * log(κ)
    term4 = (ν * n / 2) * log(2.0)
    term5 = (ν / 2) * logdet(Ψ)
    term6 = logmvgamma(n, ν / 2)
    return term1 + term2 + term3 + term4 + term5 + term6
end

getfisherinformation(::DefaultParametersSpace, ::Type{MatrixNormalWishart}, V) = (θ) -> begin
    M₀, Ψ, κ, ν = θ
    n, p = size(M₀)
    Vi   = cholinv(V)
    Ψi   = cholinv(Ψ)
    # Block structure: (M₀ block: np×np, Ψ block: n²×n², κ block: 1×1, ν block: 1×1)
    total = n * p + n^2 + 2
    F = zeros(total, total)
    @inbounds begin
        # ∂²A/∂M₀² = ν κ (V⁻¹ ⊗ Ψ⁻¹)
        F[1:(n*p), 1:(n*p)] = ν * κ * kron(Vi, Ψi)
        # ∂²A/∂Ψ²  = ν/2 (Ψ⁻¹ ⊗ Ψ⁻¹)
        F[(n*p+1):(n*p+n^2), (n*p+1):(n*p+n^2)] = (ν / 2) * kron(Ψi, Ψi)
        # ∂²A/∂κ²  = np / (2κ²)
        F[n*p+n^2+1, n*p+n^2+1] = n * p / (2κ^2)
        # ∂²A/∂ν²  = mvtrigamma(n, ν/2) / 4
        F[n*p+n^2+2, n*p+n^2+2] = mvtrigamma(n, ν / 2) / 4
    end
    return F
end

function ExponentialFamily._logpdf(ef::ExponentialFamilyDistribution{MatrixNormalWishart}, x::Tuple)
    @assert insupport(ef, x)
    return ExponentialFamily._plogpdf(ef, x)
end

function ExponentialFamily._logpdf(ef::ExponentialFamilyDistribution{MatrixNormalWishart}, xs::AbstractVector{<:Tuple})
    _logpartition = logpartition(ef)
    return map(x -> ExponentialFamily._plogpdf(ef, x, _logpartition, logbasemeasure(ef, x)), xs)
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::MatrixNormalWishart)
    M₀, Ψ, κ, ν, V = params(dist)
    n, p    = size(M₀)
    cparams = (M₀, Ψ, κ, ν)
    tuple_of_η = MeanToNatural{MatrixNormalWishart}()(cparams, V)
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, tuple_of_η)
    return ExponentialFamilyDistribution(MatrixNormalWishart, η, V)
end
