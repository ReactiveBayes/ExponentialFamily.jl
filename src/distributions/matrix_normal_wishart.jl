export MatrixNormalWishart

import Base: convert
import StatsFuns: logmvgamma
using Random

"""
    MatrixNormalWishart{T, M, U, V, N}

Models the joint distribution of a matrix `X` (n×p) and a positive-definite matrix `Y` (p×p):
- `X | Y ~ MatrixNormal(M, U, Y⁻¹)`
- `Y ~ Wishart(V, ν)`

# Fields
- `M`: Prior mean matrix (n×p).
- `U`: Row covariance matrix (n×n).
- `V`: Wishart scale matrix (p×p).
- `ν`: Wishart degrees of freedom (scalar > p−1).

# Note
Shape metadata `(n, p)` is carried through the EF interface via the `dims` field of
`ExponentialFamilyDistributionAttributes`. It is needed to reshape the flat natural-parameter
vector since it cannot be recovered uniquely from its length.

# Reference
The pdf follows the definition in the Book of Statistical Proofs (https://statproofbook.github.io/D/nw):
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

    Λl = cholinv(Ul)
    Λr = cholinv(Ur)
    Λ  = Λl + Λr
    U   = cholinv(Λ)

    ΛlMl = Λl * Ml
    ΛrMr = Λr * Mr
    rhs = ΛlMl + ΛrMr
    M = U * rhs

    Ωl = cholinv(Vl)
    Ωr = cholinv(Vr)
    Ω  = Ωl + Ωr + Ml' * ΛlMl + Mr' * ΛrMr - rhs' * M
    V   = cholinv(Ω)

    n, p = size(Ml)
    ν = νl + νr + n - p - 1

    return MatrixNormalWishart(M, U, V, ν)
end

function BayesBase.insupport(::ExponentialFamilyDistribution{MatrixNormalWishart}, x)
    return x isa Tuple && length(x) == 2 && isposdef(x[2])
end

function isproper(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, η, dims::Tuple{Int, Int})
    if any(isnan, η) || any(isinf, η)
        return false
    end
    n, p = dims
    expected_len = n * p + p^2 + 1 + n^2
    length(η) == expected_len || return false
    η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, dims)
    neg2η₄ = Symmetric(-2 * Matrix(η₄))
    isposdef(neg2η₄) || return false
    U = cholinv(neg2η₄)
    W = Symmetric(-2 * η₂ - η₁' * U * η₁)
    return η₃ > (n - 2) / 2 && isposdef(W)
end

isproper(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, η, ::Nothing) = false

function isproper(::DefaultParametersSpace, ::Type{MatrixNormalWishart}, θ, _ = nothing)
    length(θ) == 4 || return false
    M, U, V, ν = θ
    if any(x -> any(isnan, x) || any(isinf, x), (M, U, V)) || isnan(ν) || isinf(ν)
        return false
    end
    p = size(V, 1)
    return isposdef(U) && isposdef(V) && ν > p - 1
end

function (::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::NTuple{4, Any})
    M, U, V, ν = tuple_of_θ
    n, p = size(M)
    Ui = cholinv(U)
    η₁ = Ui * M
    η₂ = -one(eltype(M)) / 2 * (cholinv(V) + M' * Ui * M)
    η₃ = (ν + n - p - 1) / 2
    η₄ = -one(eltype(U)) / 2 * Ui
    return (η₁, η₂, η₃, η₄)
end

(t::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::NTuple{4, Any}, _) = t(tuple_of_θ)
(t::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::NTuple{4, Any}, _, ::Tuple{Int, Int}) = t(tuple_of_θ)
(t::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::NTuple{4, Any}, _, ::Nothing) = t(tuple_of_θ)

function (::NaturalToMean{MatrixNormalWishart})(tuple_of_η::NTuple{4, Any})
    η₁, η₂, η₃, η₄ = tuple_of_η
    n, p = size(η₁)
    U = cholinv(-2 * η₄)
    M = U * η₁
    V = cholinv(-2 * η₂ - η₁' * U * η₁)
    ν = 2 * η₃ - n + p + 1
    return (M, U, V, ν)
end

(t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::NTuple{4, Any}, _) = t(tuple_of_η)
(t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::NTuple{4, Any}, _, ::Tuple{Int, Int}) = t(tuple_of_η)
(t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::NTuple{4, Any}, _, ::Nothing) = t(tuple_of_η)

function unpack_parameters(::Type{MatrixNormalWishart}, η, dims::Tuple{Int, Int})
    "Packed layout: [vec(η₁) (n*p), vec(η₂) (p*p), η₃ (scalar), vec(η₄) (n*n)]"
    n, p = dims
    i1 = n * p
    i2 = i1 + p * p
    i3 = i2 + 1
    i4 = i3 + n * n
    @inbounds η₁ = reshape(view(η, 1:i1), n, p)
    @inbounds η₂ = reshape(view(η, (i1+1):i2), p, p)
    @inbounds η₃ = η[i3]
    @inbounds η₄ = reshape(view(η, (i3+1):i4), n, n)
    return (η₁, η₂, η₃, η₄)
end

ExponentialFamily.unpack_parameters(ef::ExponentialFamilyDistribution{MatrixNormalWishart}) =
    unpack_parameters(NaturalParametersSpace(), MatrixNormalWishart, getnaturalparameters(ef), getdims(ef))

ExponentialFamily.getfisherinformation(ef::ExponentialFamilyDistribution{MatrixNormalWishart}) =
    getfisherinformation(NaturalParametersSpace(), MatrixNormalWishart, getdims(ef))

isbasemeasureconstant(::Type{MatrixNormalWishart}) = ConstantBaseMeasure()

getbasemeasure(::Type{MatrixNormalWishart}) = (z) -> one(Float64)
getlogbasemeasure(::Type{MatrixNormalWishart}) = (z) -> zero(Float64)

function getsufficientstatistics(::Type{MatrixNormalWishart})
    """
    T₁=XY
    T₂=Y
    T₃=logdet(Y)
    T₄=XYX'
    """
    return (
        (z) -> begin
            (X, Y) = z
            X * Y
        end,
        (z) -> begin
            (_, Y) = z
            Y
        end,
        (z) -> begin
            (_, Y) = z
            logdet(Y)
        end,
        (z) -> begin
            (X, Y) = z
            X * Y * X'
        end
    )
end

getlogpartition(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, dims::Tuple{Int, Int}) =
    (η) -> begin
        η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, dims)
        n, p = dims
        η₂s = (η₂ + η₂') / 2
        η₄s = (η₄ + η₄') / 2
        U = inv(-2 * η₄s)
        V = inv(-2 * η₂s - η₁' * U * η₁)
        ν = 2 * η₃ - n + p + 1
        return (n*p/2)*log2π + (p/2)*logdet(U) + (ν/2)*logdet(V) + (ν*p/2)*log(2.0) + logmvgamma(p, ν/2)
    end

getfisherinformation(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, dims::Tuple{Int, Int}) =
    (η) -> begin
        n, p = dims
        η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, dims)

        U = cholinv(-2 * η₄)
        M = U * η₁
        V = cholinv(-2 * η₂ - η₁' * U * η₁)
        ν = 2 * η₃ - n + p + 1

        Q = M * V * M'
        R = Q + U
        N = M * V
        UQ = U + Q

        T = promote_type(eltype(η), eltype(M))
        np, pp, nn = n * p, p * p, n * n
        L = np + pp + 1 + nn
        F = zeros(T, L, L)

        o1 = 0
        o2 = np
        o3 = np + pp
        o4 = np + pp + 1
        idx1(i, a) = o1 + i + (a - 1) * n
        idx2(a, b) = o2 + a + (b - 1) * p
        idx4(i, j) = o4 + i + (j - 1) * n

        @inbounds for a′ in 1:p, i′ in 1:n, a in 1:p, i in 1:n
            F[idx1(i, a), idx1(i′, a′)] = ν * (V[a, a′] * UQ[i, i′] + N[i, a′] * N[i′, a])
        end

        @inbounds for d in 1:p, c in 1:p, b in 1:p, a in 1:p
            F[idx2(a, b), idx2(c, d)] = ν * (V[a, c] * V[b, d] + V[a, d] * V[b, c])
        end

        F[o3+1, o3+1] = mvtrigamma(p, ν / 2)

        @inbounds for l in 1:n, k in 1:n, j in 1:n, i in 1:n
            F[idx4(i, j), idx4(k, l)] =
                ν * (R[i, k] * R[j, l] + R[i, l] * R[j, k]) +
                (p - ν) * (U[i, k] * U[j, l] + U[i, l] * U[j, k])
        end

        @inbounds for b′ in 1:p, a′ in 1:p, a in 1:p, i in 1:n
            val = ν * (N[i, a′] * V[a, b′] + N[i, b′] * V[a, a′])
            F[idx1(i, a), idx2(a′, b′)] = val
            F[idx2(a′, b′), idx1(i, a)] = val
        end

        @inbounds for a in 1:p, i in 1:n
            val                 = 2 * N[i, a]
            F[idx1(i, a), o3+1] = val
            F[o3+1, idx1(i, a)] = val
        end

        @inbounds for j′ in 1:n, i′ in 1:n, a in 1:p, i in 1:n
            val = ν * (R[i, j′] * N[i′, a] + R[i, i′] * N[j′, a])
            F[idx1(i, a), idx4(i′, j′)] = val
            F[idx4(i′, j′), idx1(i, a)] = val
        end

        @inbounds for b in 1:p, a in 1:p
            val                 = 2 * V[a, b]
            F[idx2(a, b), o3+1] = val
            F[o3+1, idx2(a, b)] = val
        end

        @inbounds for j in 1:n, i in 1:n, b in 1:p, a in 1:p
            val = ν * (N[i, a] * N[j, b] + N[i, b] * N[j, a])
            F[idx2(a, b), idx4(i, j)] = val
            F[idx4(i, j), idx2(a, b)] = val
        end

        @inbounds for j in 1:n, i in 1:n
            val                 = 2 * Q[i, j]
            F[o3+1, idx4(i, j)] = val
            F[idx4(i, j), o3+1] = val
        end

        return F
    end

getlogpartition(::DefaultParametersSpace, ::Type{MatrixNormalWishart}) =
    (θ) -> begin
        M, U, V, ν = θ
        n, p = size(M)
        return (n*p/2)*log2π + (p/2)*logdet(U) + (ν/2)*logdet(V) + (ν*p/2)*log(2.0) + logmvgamma(p, ν/2)
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
    n, p = size(M)
    tuple_of_η = MeanToNatural{MatrixNormalWishart}()((M, U, V, ν))
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, tuple_of_η)
    attrs = ExponentialFamilyDistributionAttributes(
        getbasemeasure(MatrixNormalWishart),
        getsufficientstatistics(MatrixNormalWishart),
        getlogpartition(NaturalParametersSpace(), MatrixNormalWishart, (n, p)),
        nothing;
        dims = (n, p)
    )
    return ExponentialFamilyDistribution(MatrixNormalWishart, η, nothing, attrs)
end
