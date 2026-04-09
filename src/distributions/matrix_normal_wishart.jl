export MatrixNormalWishart

import Base: convert
import StatsFuns: logmvgamma
using Random

"""
    MatrixNormalWishart{T, M, U, V, N}

A Matrix Normal-Wishart distribution — the conjugate prior for the mean matrix and column
precision of a Matrix Normal distribution.

Models the joint distribution of a matrix `X` (n×p) and a positive-definite matrix `Y` (p×p):
- `X | Y ~ MatrixNormal(M, U, Y⁻¹)`
- `Y ~ Wishart(V, ν)`

# Fields
- `M`: Prior mean matrix (n×p).
- `U`: Row covariance matrix (n×n).
- `V`: Wishart scale matrix (p×p).
- `ν`: Wishart degrees of freedom (scalar > p−1).

# Note
`U` plays the role of a fixed hyperparameter (conditioner) in the exponential family interface.
When converting to `ExponentialFamilyDistribution`, `U` is stored as the conditioner.

# Reference
The pdf follows the definition in the Book of Statistical Proofs:
  MNW(X,Y | M,U,V,ν) = MN(X | M, U, Y⁻¹) W(Y | V, ν)
"""
struct MatrixNormalWishart{T, TM <: AbstractMatrix{T}, TU <: AbstractMatrix{T}, TV <: AbstractMatrix{T}, N <: Real} <:
       ContinuousMultivariateMatrixvariateDistribution
    M::TM
    U::TU
    V::TV
    ν::N

    function MatrixNormalWishart(
        M::TM,
        U::TU,
        V::TV,
        ν::N
    ) where {T, TM <: AbstractMatrix{T}, TU <: AbstractMatrix{T}, TV <: AbstractMatrix{T}, N <: Real}
        new{T, TM, TU, TV, N}(M, U, V, ν)
    end

    function MatrixNormalWishart(
        M::TM,
        U::TU,
        V::TV,
        ν::N
    ) where {T1, T2, T3, TM <: AbstractMatrix{T1}, TU <: AbstractMatrix{T2}, TV <: AbstractMatrix{T3}, N <: Real}
        T = promote_type(T1, T2, T3)
        M_new = convert(AbstractMatrix{T}, M)
        U_new = convert(AbstractMatrix{T}, U)
        V_new = convert(AbstractMatrix{T}, V)
        return new{T, typeof(M_new), typeof(U_new), typeof(V_new), N}(M_new, U_new, V_new, ν)
    end
end

Base.eltype(::MatrixNormalWishart{T}) where {T} = T
Base.size(d::MatrixNormalWishart) = size(d.M)

BayesBase.params(d::MatrixNormalWishart)   = (d.M, d.U, d.V, d.ν)
BayesBase.mean(d::MatrixNormalWishart)     = (d.M, d.ν * d.V)
BayesBase.dof(d::MatrixNormalWishart)      = d.ν
BayesBase.location(d::MatrixNormalWishart) = d.M

# U is the conditioner; the "free" parameters are (M, V, ν).
ExponentialFamily.separate_conditioner(::Type{MatrixNormalWishart}, params) =
    ((params[1], params[3], params[4]), params[2])

ExponentialFamily.join_conditioner(::Type{MatrixNormalWishart}, cparams, U) =
    (cparams[1], U, cparams[2], cparams[3])

function BayesBase.pdf(dist::MatrixNormalWishart, x::Tuple)
    (X, Y) = x
    M, U, V, ν = params(dist)
    return pdf(MatrixNormal(M, U, cholinv(Y)), X) * pdf(Wishart(ν, V), Y)
end

BayesBase.logpdf(dist::MatrixNormalWishart, x::Tuple) = log(pdf(dist, x))

function BayesBase.rand(rng::AbstractRNG, dist::MatrixNormalWishart{T}) where {T}
    M, U, V, ν = params(dist)
    Y = rand(rng, Wishart(ν, V))
    X = rand(rng, MatrixNormal(M, U, cholinv(Y)))
    return (X, Y)
end

function BayesBase.rand(rng::AbstractRNG, dist::MatrixNormalWishart{T}, nsamples::Int) where {T}
    return [rand(rng, dist) for _ in 1:nsamples]
end

BayesBase.default_prod_rule(::Type{<:MatrixNormalWishart}, ::Type{<:MatrixNormalWishart}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MatrixNormalWishart, right::MatrixNormalWishart)
    Ml, Ul, Vl, νl = params(left)
    Mr, Ur, Vr, νr = params(right)
    # Assumes same conditioner Ul == Ur = U
    U  = Ul
    Ui = cholinv(U)
    M  = Ml + Mr
    Vl_inv = cholinv(Vl)
    Vr_inv = cholinv(Vr)
    cross  = Ml' * Ui * Mr
    V  = cholinv(Vl_inv + Vr_inv - cross - cross')
    n, p = size(Ml)
    ν  = νl + νr + n - p - 1
    return MatrixNormalWishart(M, U, V, ν)
end

function BayesBase.insupport(::ExponentialFamilyDistribution{MatrixNormalWishart}, x)
    return x isa Tuple && length(x) == 2 && isposdef(x[2])
end

function isproper(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, η, U)
    if isnothing(U) || any(isnan, η) || any(isinf, η)
        return false
    end
    η₁, η₂, η₃ = unpack_parameters(MatrixNormalWishart, η, U)
    n = size(η₁, 1)
    W = Symmetric(-2η₂ - η₁' * U * η₁)
    return η₃ > (n - 2) / 2 && isposdef(W)
end

function isproper(::DefaultParametersSpace, ::Type{MatrixNormalWishart}, θ, U)
    if isnothing(U)
        return false
    end
    M, V, ν = θ
    if any(x -> any(isnan, x) || any(isinf, x), (M, V)) || isnan(ν) || isinf(ν)
        return false
    end
    p = size(V, 1)
    return isposdef(V) && ν > p - 1
end

function (::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::Tuple{Any, Any, Any}, U)
    M, V, ν = tuple_of_θ
    n, p = size(M)
    Ui   = cholinv(U)
    η₁   = Ui * M                               # n×p
    η₂   = -1 / 2 * (cholinv(V) + M' * Ui * M) # p×p
    η₃   = (ν + n - p - 1) / 2                 # scalar
    return (η₁, η₂, η₃)
end

function (::NaturalToMean{MatrixNormalWishart})(tuple_of_η::Tuple{Any, Any, Any}, U)
    η₁, η₂, η₃ = tuple_of_η
    n, p = size(η₁)
    M  = U * η₁                                  # n×p
    V  = cholinv(-2η₂ - η₁' * U * η₁)           # p×p
    ν  = 2η₃ - n + p + 1                        # scalar
    return (M, V, ν)
end

function (t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::Tuple{Any, Any, Any}, U, ::Tuple{Int, Int})
    return t(tuple_of_η, U)
end

function (t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::Tuple{Any, Any, Any}, U, ::Nothing)
    return t(tuple_of_η, U)
end

# Packed layout: [vec(η₁) (n×p), vec(η₂) (p×p), η₃]   length = np + p² + 1
# Conditioner U is n×n, so n = size(U,1).
# p satisfies: p² + n*p − (length(η)−1) = 0  →  p = (−n + √(n²+4L)) / 2  where L = length(η)-1
function unpack_parameters(::Type{MatrixNormalWishart}, η, U)
    n = size(U, 1)
    L = length(η) - 1
    p = Int(round((-n + sqrt(n^2 + 4L)) / 2))
    @inbounds η₁ = reshape(view(η, 1:(n * p)), n, p)
    @inbounds η₂ = reshape(view(η, (n * p + 1):(n * p + p^2)), p, p)
    @inbounds η₃ = η[n * p + p^2 + 1]
    return η₁, η₂, η₃
end

isbasemeasureconstant(::Type{MatrixNormalWishart}) = NonConstantBaseMeasure()

getbasemeasure(::Type{MatrixNormalWishart}, U) = let Ui = cholinv(U)
    (z) -> begin (X, Y) = z; exp(-1 / 2 * tr(X' * Ui * X * Y)) end
end

getlogbasemeasure(::Type{MatrixNormalWishart}, U) = let Ui = cholinv(U)
    (z) -> begin (X, Y) = z; -1 / 2 * tr(X' * Ui * X * Y) end
end

function getsufficientstatistics(::Type{MatrixNormalWishart}, U)
    return (
        (z) -> begin (X, Y) = z; X * Y end,         # T₁ = XY (n×p)
        (z) -> begin (_, Y) = z; Y end,              # T₂ = Y  (p×p)
        (z) -> begin (_, Y) = z; logdet(Y) end       # T₃ = logdet(Y)
    )
end

getlogpartition(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, U) = (η) -> begin
    η₁, η₂, η₃ = unpack_parameters(MatrixNormalWishart, η, U)
    n, p = size(η₁)
    ν  = 2η₃ - n + p + 1
    V  = cholinv(-2η₂ - η₁' * U * η₁)
    term1 = (n * p / 2) * log2π
    term2 = (p / 2) * logdet(U)
    term3 = (ν / 2) * logdet(V)
    term4 = (ν * p / 2) * log(2.0)
    term5 = logmvgamma(p, ν / 2)
    return term1 + term2 + term3 + term4 + term5
end

getlogpartition(::DefaultParametersSpace, ::Type{MatrixNormalWishart}, U) = (θ) -> begin
    M, V, ν = θ
    n, p = size(M)
    term1 = (n * p / 2) * log2π
    term2 = (p / 2) * logdet(U)
    term3 = (ν / 2) * logdet(V)
    term4 = (ν * p / 2) * log(2.0)
    term5 = logmvgamma(p, ν / 2)
    return term1 + term2 + term3 + term4 + term5
end

getfisherinformation(::DefaultParametersSpace, ::Type{MatrixNormalWishart}, U) = (θ) -> begin
    M, V, ν = θ
    n, p = size(M)
    Ui   = cholinv(U)
    Vi   = cholinv(V)
    # Block structure: (M block: np×np, V block: p²×p², ν block: 1×1)
    total = n * p + p^2 + 1
    F = zeros(total, total)
    @inbounds begin
        # ∂²A/∂M² = ν (V⁻¹ ⊗ U⁻¹)
        F[1:(n * p), 1:(n * p)] = ν * kron(Vi, Ui)
        # ∂²A/∂V²  = ν/2 (V⁻¹ ⊗ V⁻¹)
        F[(n * p + 1):(n * p + p^2), (n * p + 1):(n * p + p^2)] = (ν / 2) * kron(Vi, Vi)
        # ∂²A/∂ν²  = mvtrigamma(p, ν/2) / 4
        F[n * p + p^2 + 1, n * p + p^2 + 1] = mvtrigamma(p, ν / 2) / 4
    end
    return F
end

BayesBase.default_prod_rule(
    ::Type{<:ExponentialFamilyDistribution{MatrixNormalWishart}},
    ::Type{<:ExponentialFamilyDistribution{MatrixNormalWishart}}
) = PreserveTypeProd(ExponentialFamilyDistribution{MatrixNormalWishart})

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution{MatrixNormalWishart}},
    left::ExponentialFamilyDistribution{MatrixNormalWishart},
    right::ExponentialFamilyDistribution{MatrixNormalWishart}
)
    if !isapprox(getconditioner(left), getconditioner(right))
        error("MatrixNormalWishart product requires matching conditioners (row covariance U)")
    end
    F = promote_type(eltype(getnaturalparameters(left)), eltype(getnaturalparameters(right)))
    container = similar(left, F)
    return BayesBase.prod!(container, left, right)
end

function BayesBase.prod!(
    container::ExponentialFamilyDistribution{MatrixNormalWishart},
    left::ExponentialFamilyDistribution{MatrixNormalWishart},
    right::ExponentialFamilyDistribution{MatrixNormalWishart}
)
    if !isapprox(getconditioner(left), getconditioner(right)) || !isapprox(getconditioner(container), getconditioner(left))
        error("MatrixNormalWishart product requires matching conditioners (row covariance U)")
    end
    map!(+, getnaturalparameters(container), getnaturalparameters(left), getnaturalparameters(right))
    return container
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
    M, U, V, ν = params(dist)
    cparams    = (M, V, ν)
    tuple_of_η = MeanToNatural{MatrixNormalWishart}()(cparams, U)
    η          = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, tuple_of_η)
    return ExponentialFamilyDistribution(MatrixNormalWishart, η, U)
end
