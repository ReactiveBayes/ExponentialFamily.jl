export MatrixNormal

import Distributions: entropy, distrname, AbstractMvNormal
import Base: convert
import ForwardDiff

using DomainSets
using LinearAlgebra

Distributions.distrname(::MatrixNormal) = "MatrixNormal"

Base.eltype(::MatrixNormal{T}) where {T} = T
Base.precision(d::MatrixNormal) = invcov(d)

covmats(d::MatrixNormal) = (d.U, d.V)
BayesBase.invcov(d::MatrixNormal) = (cholinv(d.U), cholinv(d.V))
BayesBase.vague(::Type{<:MatrixNormal}, dims::Tuple{Int, Int}) =
    MatrixNormal(zeros(Float64, dims), diagm(ones(dims[1])), diagm(ones(dims[2])))

function kronecker_factor(M::AbstractMatrix, n::Int, p::Int)
    # M = V⁻¹ ⊗ U⁻¹  (passed as -2η2)
    # Reshape so that the (i,j),(k,l) element of M becomes the (i*p+k, j*n+l) element of a rank-1 matrix
    R = reshape(permutedims(reshape(M, n, p, n, p), (2, 4, 1, 3)), p^2, n^2)
    F = svd(R)
    sqrts = sqrt(F.S[1])
    Vi = reshape(F.U[:, 1] * sqrts, p, p)   # ≈ V⁻¹
    Ui = reshape(F.V[:, 1] * sqrts, n, n)   # ≈ U⁻¹

    # Fix sign ambiguity: both U⁻¹ and V⁻¹ must be positive definite
    if tr(Ui) < 0
        Ui = -Ui
        Vi = -Vi
    end

    return Ui, Vi
end

getsupport(::MatrixNormal) = ProductDomain(VectorDomain{FullSpace}, VectorDomain{FullSpace})

function Base.convert(::Type{MvNormalMeanCovariance}, d::MatrixNormal)
    U, V = covmats(d)
    return MvNormalMeanCovariance(vec(mean(d)), Distributions.PDMats.PDMat(kron(V, U)))
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::MatrixNormal)
    M, U, V = params(dist)
    n, p = size(M)
    tuple_of_η = MeanToNatural{MatrixNormal}()((M, U, V))
    η = pack_parameters(NaturalParametersSpace(), MatrixNormal, tuple_of_η)
    attrs = ExponentialFamilyDistributionAttributes(
        getbasemeasure(MatrixNormal),
        getsufficientstatistics(MatrixNormal),
        getlogpartition(NaturalParametersSpace(), MatrixNormal, (n, p)),
        nothing;
        dims = (n, p)
    )
    return ExponentialFamilyDistribution(MatrixNormal, η, nothing, attrs)
end

BayesBase.default_prod_rule(::Type{MatrixNormal}, ::Type{MatrixNormal}) = PreserveTypeProd(MvNormalMeanCovariance)

function BayesBase.prod(::PreserveTypeProd{MvNormalMeanCovariance}, left::MatrixNormal, right::MatrixNormal)
    Ui_l, Vi_l = invcov(left)
    Ui_r, Vi_r = invcov(right)

    # Combined precision on vec(X): Λ = (V_l⁻¹⊗U_l⁻¹) + (V_r⁻¹⊗U_r⁻¹)
    Λ = kron(Vi_l, Ui_l) + kron(Vi_r, Ui_r)

    # Combined weighted mean: ξ = U_l⁻¹ M_l V_l⁻¹ + U_r⁻¹ M_r V_r⁻¹ (vectorised)
    ξ = vec(Ui_l * mean(left) * Vi_l) + vec(Ui_r * mean(right) * Vi_r)
    Σ = cholinv(Λ)

    return MvNormalMeanCovariance(Σ * ξ, Σ)
end

function (::MeanToNatural{MatrixNormal})(tuple_of_θ::Tuple{Any, Any, Any})
    (M, U, V) = tuple_of_θ
    Ui = cholinv(U)
    Vi = cholinv(V)
    η1 = vec(Ui*M*Vi)
    η2 = -1/2*kron(Vi, Ui)
    return (η1, η2)
end

function (::NaturalToMean{MatrixNormal})(tuple_of_η::Tuple{Any, Any}, ::Nothing, dims::Tuple{Int, Int})
    (η1, η2) = tuple_of_η
    n, p = dims
    Ui, Vi = kronecker_factor(-2η2, n, p)
    U = cholinv(Ui)
    V = cholinv(Vi)
    M = U*reshape(η1, n, p)*V
    return (M, U, V)
end

# Mean parametrization

function Distributions.entropy(d::MatrixNormal)
    (M, U, V) = params(d)
    n, p = size(d)
    return n*p/2*log(2*π) + p/2*logdet(U) + n/2*logdet(V) + n*p/2
end

getlogpartition(::DefaultParametersSpace, ::Type{MatrixNormal}) = (Θ) -> begin
    (M, U, V) = Θ
    n, p = size(M)
    return n*p/2*log(2*π) + p/2*logdet(U) + n/2*logdet(V)
end

getgradlogpartition(::DefaultParametersSpace, ::Type{MatrixNormal}) = (Θ) -> begin
    (M, U, V) = Θ
    n, p = size(M)
    return (zeros(n, p), p/2*cholinv(U), n/2*cholinv(V))
end

getfisherinformation(::DefaultParametersSpace, ::Type{MatrixNormal}) = (Θ) -> begin
    (M, U, V) = Θ
    n, p = size(M)
    Ui = cholinv(U)
    Vi = cholinv(V)
    return (kron(Ui, Vi), n/2*kron(Ui, Ui), p/2*kron(Vi, Vi))
end

# Natural parameterization

isproper(::NaturalParametersSpace, ::Type{MatrixNormal}, η, conditioner) = isnothing(conditioner) && all(x -> !any(isinf, x) && !any(isnan, x), η)
isproper(::DefaultParametersSpace, ::Type{MatrixNormal}, θ, conditioner) = isnothing(conditioner) && all(x -> !any(isinf, x) && !any(isnan, x), θ)

isbasemeasureconstant(::Type{MatrixNormal}) = ConstantBaseMeasure()
getbasemeasure(::Type{MatrixNormal}) = (x) -> oneunit(x)

getnaturalparameters(::DefaultParametersSpace, ::Type{MatrixNormal}) = (θ) -> begin
    (M, U, V) = θ
    Ui = cholinv(U)
    Vi = cholinv(V)
    η1 = vec(Ui*M*Vi)
    η2 = -1/2*kron(Vi, Ui)
    return (η1, η2)
end

getsufficientstatistics(::Type{MatrixNormal}) = (
    (X) -> vec(X),
    (X) -> vec(X) * vec(X)'
)

function unpack_parameters(::Type{MatrixNormal}, η, dims::Tuple{Int, Int})
    n, p = dims
    np = n * p
    i1 = np
    i2 = i1 + np * np
    @inbounds η₁ = view(η, 1:i1)
    @inbounds η₂ = reshape(view(η, (i1+1):i2), np, np)
    return (η₁, η₂)
end

unpack_parameters(::Union{DefaultParametersSpace, NaturalParametersSpace}, ::Type{MatrixNormal}, packed, dims::Tuple{Int, Int}) =
    unpack_parameters(MatrixNormal, packed, dims)

ExponentialFamily.unpack_parameters(ef::ExponentialFamilyDistribution{MatrixNormal}) =
    unpack_parameters(NaturalParametersSpace(), MatrixNormal, getnaturalparameters(ef), getdims(ef))

function isproper(::NaturalParametersSpace, ::Type{MatrixNormal}, η, dims::Tuple{Int, Int})
    n, p = dims
    np = n * p
    length(η) == np + np * np || return false
    (any(isnan, η) || any(isinf, η)) && return false
    _, η₂ = unpack_parameters(MatrixNormal, η, dims)
    return isposdef(Symmetric(-2 * Matrix(η₂)))
end

ExponentialFamily.isproper(ef::ExponentialFamilyDistribution{MatrixNormal}) =
    isproper(NaturalParametersSpace(), MatrixNormal, getnaturalparameters(ef), getdims(ef))

BayesBase.insupport(::ExponentialFamilyDistribution{MatrixNormal}, ::AbstractMatrix) = true

getlogpartition(::NaturalParametersSpace, ::Type{MatrixNormal}, dims::Tuple{Int, Int}) =
    (η) -> begin
        n, p = dims
        np = n * p
        η₁, η₂ = unpack_parameters(MatrixNormal, η, dims)
        K = Symmetric(-2 * (η₂ + η₂') / 2)
        Kinv = inv(K)
        return (np/2) * log(2π) - (1/2) * logdet(K) + (1/2) * dot(η₁, Kinv, η₁)
    end

getgradlogpartition(::NaturalParametersSpace, ::Type{MatrixNormal}, dims::Tuple{Int, Int}) =
    (η) -> begin
        η₁, η₂ = unpack_parameters(MatrixNormal, η, dims)
        K = Symmetric(-2 * (η₂ + η₂') / 2)
        Kinv = inv(K)
        m = Kinv * η₁
        grad_T2 = Kinv + m * m'
        return vcat(m, vec(grad_T2))
    end

ExponentialFamily.getgradlogpartition(ef::ExponentialFamilyDistribution{MatrixNormal}) =
    getgradlogpartition(NaturalParametersSpace(), MatrixNormal, getdims(ef))

getfisherinformation(::NaturalParametersSpace, ::Type{MatrixNormal}, dims::Tuple{Int, Int}) =
    (η) -> getfisherinformation(NaturalParametersSpace(), MvNormalMeanCovariance)(η)

ExponentialFamily.getfisherinformation(ef::ExponentialFamilyDistribution{MatrixNormal}) =
    getfisherinformation(NaturalParametersSpace(), MatrixNormal, getdims(ef))
