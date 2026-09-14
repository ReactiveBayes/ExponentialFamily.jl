export MatrixNormalWishart

import Base: convert
import StatsFuns: logmvgamma
using Random

# --- svec/smat helpers ---
# svec packs a symmetric p×p matrix into a length-p(p+1)/2 vector with √2 scaling
# on off-diagonal entries, so that ⟨svec(A), svec(B)⟩ = tr(A*B) for symmetric A, B.
# Ordering: column-major upper triangle: (1,1), (1,2), (2,2), (1,3), (2,3), (3,3), ...
# Equivalently, idx(a, b) = a + b*(b-1)÷2 for 1 ≤ a ≤ b ≤ p.
@inline _mnw_svec_len(p::Int) = (p * (p + 1)) ÷ 2
@inline _mnw_svec_idx(a::Int, b::Int) = a ≤ b ? a + (b * (b - 1)) ÷ 2 : b + (a * (a - 1)) ÷ 2

function _mnw_svec(S::AbstractMatrix{T}) where {T}
    p = size(S, 1)
    v = Vector{T}(undef, _mnw_svec_len(p))
    s2 = sqrt(T(2))
    idx = 1
    @inbounds for b in 1:p
        for a in 1:(b-1)
            v[idx] = s2 * S[a, b]
            idx += 1
        end
        v[idx] = S[b, b]
        idx += 1
    end
    return v
end

function _mnw_smat(v::AbstractVector{T}, p::Int) where {T}
    S = Matrix{T}(undef, p, p)
    s2 = sqrt(T(2))
    idx = 1
    @inbounds for b in 1:p
        for a in 1:(b-1)
            val = v[idx] / s2
            S[a, b] = val
            S[b, a] = val
            idx += 1
        end
        S[b, b] = v[idx]
        idx += 1
    end
    return S
end

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
    expected_len = n * p + _mnw_svec_len(p) + 1 + _mnw_svec_len(n)
    length(η) == expected_len || return false
    η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, dims)
    η₂m = _mnw_smat(η₂, p)
    η₄m = _mnw_smat(η₄, n)
    neg2η₄ = Symmetric(-2 * η₄m)
    isposdef(neg2η₄) || return false
    U = cholinv(neg2η₄)
    W = Symmetric(-2 * η₂m - η₁' * U * η₁)
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
    return (η₁, _mnw_svec(η₂), η₃, _mnw_svec(η₄))
end

(t::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::NTuple{4, Any}, ::Nothing) = t(tuple_of_θ)
(t::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::NTuple{4, Any}, _) = t(tuple_of_θ)
(t::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::NTuple{4, Any}, _, ::Tuple{Int, Int}) = t(tuple_of_θ)
(t::MeanToNatural{MatrixNormalWishart})(tuple_of_θ::NTuple{4, Any}, _, ::Nothing) = t(tuple_of_θ)

function (::NaturalToMean{MatrixNormalWishart})(tuple_of_η::NTuple{4, Any})
    η₁, η₂, η₃, η₄ = tuple_of_η
    n, p = size(η₁)
    η₂m = _mnw_smat(η₂, p)
    η₄m = _mnw_smat(η₄, n)
    U = cholinv(-2 * η₄m)
    M = U * η₁
    V = cholinv(-2 * η₂m - η₁' * U * η₁)
    ν = 2 * η₃ - n + p + 1
    return (M, U, V, ν)
end

(t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::NTuple{4, Any}, _) = t(tuple_of_η)
(t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::NTuple{4, Any}, _, ::Tuple{Int, Int}) = t(tuple_of_η)
(t::NaturalToMean{MatrixNormalWishart})(tuple_of_η::NTuple{4, Any}, _, ::Nothing) = t(tuple_of_η)

function unpack_parameters(::Type{MatrixNormalWishart}, η, dims::Tuple{Int, Int})
    "Packed layout: [vec(η₁) (n*p), svec(η₂) (p(p+1)/2), η₃ (scalar), svec(η₄) (n(n+1)/2)]"
    n, p = dims
    ph = _mnw_svec_len(p)
    nh = _mnw_svec_len(n)
    i1 = n * p
    i2 = i1 + ph
    i3 = i2 + 1
    i4 = i3 + nh
    @inbounds η₁ = reshape(view(η, 1:i1), n, p)
    @inbounds η₂ = view(η, (i1+1):i2)
    @inbounds η₃ = η[i3]
    @inbounds η₄ = view(η, (i3+1):i4)
    return (η₁, η₂, η₃, η₄)
end

ExponentialFamily.unpack_parameters(ef::ExponentialFamilyDistribution{MatrixNormalWishart}) =
    unpack_parameters(NaturalParametersSpace(), MatrixNormalWishart, getnaturalparameters(ef), getdims(ef))

ExponentialFamily.getfisherinformation(ef::ExponentialFamilyDistribution{MatrixNormalWishart}) =
    getfisherinformation(NaturalParametersSpace(), MatrixNormalWishart, getdims(ef))

ExponentialFamily.getgradlogpartition(ef::ExponentialFamilyDistribution{MatrixNormalWishart}) =
    getgradlogpartition(NaturalParametersSpace(), MatrixNormalWishart, getdims(ef))

ExponentialFamily.isproper(ef::ExponentialFamilyDistribution{MatrixNormalWishart}) =
    isproper(NaturalParametersSpace(), MatrixNormalWishart, getnaturalparameters(ef), getdims(ef))

isbasemeasureconstant(::Type{MatrixNormalWishart}) = ConstantBaseMeasure()

getbasemeasure(::Type{MatrixNormalWishart}) = (z) -> one(Float64)
getlogbasemeasure(::Type{MatrixNormalWishart}) = (z) -> zero(Float64)

function getsufficientstatistics(::Type{MatrixNormalWishart})
    """
    T₁ = XY                 (n×p)
    T₂ = svec(Y)            (p(p+1)/2)
    T₃ = logdet(Y)          (scalar)
    T₄ = svec(XYX')         (n(n+1)/2)
    The √2 off-diagonal scaling in svec ensures ⟨svec(η), svec(T)⟩ = tr(η*T)
    for symmetric η, T, so the natural-parameter inner product is preserved.
    """
    return (
        (z) -> begin
            (X, Y) = z
            X * Y
        end,
        (z) -> begin
            (_, Y) = z
            _mnw_svec(Y)
        end,
        (z) -> begin
            (_, Y) = z
            logdet(Y)
        end,
        (z) -> begin
            (X, Y) = z
            _mnw_svec(X * Y * X')
        end
    )
end

getlogpartition(::DefaultParametersSpace, ::Type{MatrixNormalWishart}) =
    (θ) -> begin
        M, U, V, ν = θ
        n, p = size(M)
        return (n*p/2)*log2π + (p/2)*logdet(U) + (ν/2)*logdet(V) + (ν*p/2)*log(2.0) + logmvgamma(p, ν/2)
    end

getlogpartition(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, dims::Tuple{Int, Int}) =
    (η) -> begin
        n, p = dims
        η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, dims)
        η₂m = _mnw_smat(η₂, p)
        η₄m = _mnw_smat(η₄, n)
        U = inv(-2 * η₄m)
        V = inv(-2 * η₂m - η₁' * U * η₁)
        ν = 2 * η₃ - n + p + 1
        return (n*p/2)*log2π + (p/2)*logdet(U) + (ν/2)*logdet(V) + (ν*p/2)*log(2.0) + logmvgamma(p, ν/2)
    end

getgradlogpartition(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, dims::Tuple{Int, Int}) =
    (η) -> begin
        n, p = dims
        η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, dims)

        η₂m = _mnw_smat(η₂, p)
        η₄m = _mnw_smat(η₄, n)
        U = inv(-2 * η₄m)
        M = U * η₁
        V = inv(-2 * η₂m - η₁' * U * η₁)
        ν = 2 * η₃ - n + p + 1

        grad_T1 = ν * (M * V)
        grad_T2 = ν * V
        grad_T3 = mvdigamma(ν / 2, p) + p * log(2.0) + logdet(V)
        grad_T4 = ν * (M * V * M') + p * U

        return vcat(vec(grad_T1), _mnw_svec(grad_T2), grad_T3, _mnw_svec(grad_T4))
    end

getfisherinformation(::NaturalParametersSpace, ::Type{MatrixNormalWishart}, dims::Tuple{Int, Int}) =
    (η) -> begin
        # F = Cov(T) in the svec'd packed layout. With c(a,b) = (a == b ? 1 : √2),
        # Cov(svec(S)_{(a,b)}, svec(S')_{(c,d)}) = c(a,b)·c(c,d)·Cov(S_{a,b}, S'_{c,d}).
        # Let N = MV, Q = MVM', R = U + Q.
        n, p = dims
        η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, dims)
        η₂m = _mnw_smat(η₂, p)
        η₄m = _mnw_smat(η₄, n)
        U = inv(-2 * η₄m)
        M = U * η₁
        V = inv(-2 * η₂m - η₁' * U * η₁)
        ν = 2 * η₃ - n + p + 1

        N = M * V
        Q = M * V * M'
        R = U + Q

        T = promote_type(eltype(η), Float64)
        s2 = sqrt(T(2))
        c2 = (a, b) -> (a == b ? one(T) : s2)

        np = n * p
        ph = _mnw_svec_len(p)
        nh = _mnw_svec_len(n)
        o2 = np
        o3 = np + ph + 1
        o4 = np + ph + 1
        total = np + ph + 1 + nh
        F = zeros(T, total, total)

        idx1 = (i, a) -> i + (a - 1) * n
        idx2 = (a, b) -> o2 + _mnw_svec_idx(a, b)
        idx4 = (i, j) -> o4 + _mnw_svec_idx(i, j)

        # (1,1): Cov(T₁_{ia}, T₁_{jb}) = ν (V_{ab} R_{ij} + N_{ib} N_{ja})
        for a in 1:p, i in 1:n, b in 1:p, j in 1:n
            F[idx1(i, a), idx1(j, b)] = ν * (V[a, b] * R[i, j] + N[i, b] * N[j, a])
        end

        # (1,2) / (2,1): c(c,d) · ν (N_{ic} V_{ad} + N_{id} V_{ac}), c ≤ d
        for a in 1:p, i in 1:n, d in 1:p, c in 1:d
            val = c2(c, d) * ν * (N[i, c] * V[a, d] + N[i, d] * V[a, c])
            F[idx1(i, a), idx2(c, d)] = val
            F[idx2(c, d), idx1(i, a)] = val
        end

        # (1,3) / (3,1): 2 N_{ia}
        for a in 1:p, i in 1:n
            val = 2 * N[i, a]
            F[idx1(i, a), o3] = val
            F[o3, idx1(i, a)] = val
        end

        # (1,4) / (4,1): c(k,l) · ν (R_{ik} N_{la} + R_{il} N_{ka}), k ≤ l
        for a in 1:p, i in 1:n, l in 1:n, k in 1:l
            val = c2(k, l) * ν * (R[i, k] * N[l, a] + R[i, l] * N[k, a])
            F[idx1(i, a), idx4(k, l)] = val
            F[idx4(k, l), idx1(i, a)] = val
        end

        # (2,2): c(a,b)·c(c,d) · ν (V_{ac} V_{bd} + V_{ad} V_{bc}), a ≤ b and c ≤ d
        for b in 1:p, a in 1:b, d in 1:p, c in 1:d
            F[idx2(a, b), idx2(c, d)] = c2(a, b) * c2(c, d) * ν * (V[a, c] * V[b, d] + V[a, d] * V[b, c])
        end

        # (2,3) / (3,2): c(a,b) · 2 V_{ab}, a ≤ b
        for b in 1:p, a in 1:b
            val = c2(a, b) * 2 * V[a, b]
            F[idx2(a, b), o3] = val
            F[o3, idx2(a, b)] = val
        end

        # (2,4) / (4,2): c(a,b)·c(k,l) · ν (N_{ka} N_{lb} + N_{kb} N_{la}), a ≤ b and k ≤ l
        for b in 1:p, a in 1:b, l in 1:n, k in 1:l
            val = c2(a, b) * c2(k, l) * ν * (N[k, a] * N[l, b] + N[k, b] * N[l, a])
            F[idx2(a, b), idx4(k, l)] = val
            F[idx4(k, l), idx2(a, b)] = val
        end

        # (3,3): mvtrigamma(p, ν/2)
        F[o3, o3] = mvtrigamma(p, ν / 2)

        # (3,4) / (4,3): c(k,l) · 2 Q_{kl}, k ≤ l
        for l in 1:n, k in 1:l
            val = c2(k, l) * 2 * Q[k, l]
            F[o3, idx4(k, l)] = val
            F[idx4(k, l), o3] = val
        end

        # (4,4): c(i,j)·c(k,l) · [ν(R_{ik}R_{jl} + R_{il}R_{jk}) + (p-ν)(U_{ik}U_{jl} + U_{il}U_{jk})],
        # for i ≤ j and k ≤ l
        for j in 1:n, i in 1:j, l in 1:n, k in 1:l
            F[idx4(i, j), idx4(k, l)] = c2(i, j) * c2(k, l) * (
                ν * (R[i, k]*R[j, l] + R[i, l]*R[j, k]) + (p - ν) * (U[i, k]*U[j, l] + U[i, l]*U[j, k])
            )
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
