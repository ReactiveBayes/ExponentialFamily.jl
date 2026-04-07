export MatrixNormal

import Distributions: entropy,distrname,AbstractMvNormal
import Base: convert

using DomainSets
using LinearAlgebra


Distributions.distrname(::MatrixNormal) = "MatrixNormal"

Base.eltype(::MatrixNormal{T}) where {T} = T
Base.precision(d::MatrixNormal) = invcov(d)

covmats(d::MatrixNormal) = (d.U,d.V)
BayesBase.invcov(d::MatrixNormal) = (cholinv(d.U),cholinv(d.V))
BayesBase.vague(::Type{<:MatrixNormal}, dims::Tuple{Int,Int}) =
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

getsupport(::MatrixNormal) = ProductDomain(VectorDomain{FullSpace},VectorDomain{FullSpace})

function Base.convert(::Type{MvNormalMeanCovariance}, d::MatrixNormal)
    U,V = covmats(d)
    return MvNormalMeanCovariance(vec(mean(d)), Distributions.PDMats.PDMat(kron(V,U)))
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

function (::MeanToNatural{MatrixNormal})(tuple_of_θ::Tuple{Any,Any,Any})
    (M,U,V) = tuple_of_θ
    Ui = cholinv(U)
    Vi = cholinv(V)
    η1 = vec(Ui*M*Vi)
    η2 = -1/2*kron(Vi,Ui)
    return (η1,η2)
end

function (::NaturalToMean{MatrixNormal})(tuple_of_η::Tuple{Any,Any}, ::Nothing, dims::Tuple{Int,Int})
    (η1, η2) = tuple_of_η
    n,p = dims
    Ui,Vi = kronecker_factor(-2η2,n,p)
    U = cholinv(Ui)
    V = cholinv(Vi)
    M = U*reshape(η1,n,p)*V
    return (M,U,V)
end

# Natural parameterization

isproper(::NaturalParametersSpace, ::Type{MatrixNormal}, η, conditioner) = isnothing(conditioner) && all(x -> !any(isinf, x) && !any(isnan, x), η)
isproper(::DefaultParametersSpace, ::Type{MatrixNormal}, θ, conditioner) = isnothing(conditioner) && !any(isinf, θ) && !any(isnan, θ)

isbasemeasureconstant(::Type{MatrixNormal}) = ConstantBaseMeasure()
getbasemeasure(::Type{MatrixNormal}) = (x) -> oneunit(x)

getnaturalparameters(::DefaultParametersSpace, ::Type{MatrixNormal}) = (θ) -> begin
    (M,U,V) = θ
    Ui = cholinv(U)
    Vi = cholinv(V)
    η1 = vec(Ui*M*Vi)
    η2 = -1/2*kron(Vi,Ui)
    return (η1,η2)  
end

getsufficientstatistics(::Type{MatrixNormal}) = (X) -> begin
    T1 = vec(X)
    T2 = vec(X)*vec(X)'
    return (T1,T2)
end

# Mean parametrization

function Distributions.entropy(d::MatrixNormal)
    (M,U,V) = params(d)
    n,p = size(d)
    return n*p/2*log(2*π) + p/2*logdet(U) + n/2*logdet(V) + n*p/2
end

getlogpartition(::DefaultParametersSpace, ::Type{MatrixNormal}) = (Θ) -> begin
    (M,U,V) = Θ
    n,p = size(M)
    return n*p/2*log(2*π) + p/2*logdet(U) + n/2*logdet(V)
end

getgradlogpartition(::DefaultParametersSpace, ::Type{MatrixNormal}) = (Θ) -> begin
    (M,U,V) = Θ
    n,p = size(M)
    return (zeros(n,p), p/2*cholinv(U), n/2*cholinv(V))
end

getfisherinformation(::DefaultParametersSpace, ::Type{MatrixNormal}) = (Θ) -> begin
    (M,U,V) = Θ
    n,p = size(M)
    Ui = cholinv(U)
    Vi = cholinv(V)
    return (kron(Ui,Vi), n/2*kron(Ui,Ui), p/2*kron(Vi,Vi)) 
end