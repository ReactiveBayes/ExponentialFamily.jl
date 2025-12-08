export MatrixNormal

import Distributions: cov,entropy,distrname
import Base: convert

using DomainSets
using LinearAlgebra


Distributions.distrname(::MatrixNormal) = "MatrixNormal"

Base.eltype(::MatrixNormal{T}) where {T} = T
Base.precision(d::MatrixNormal) = invcov(d)

BayesBase.cov(d::MatrixNormal) = (d.U,d.V)
BayesBase.invcov(d::MatrixNormal) = (cholinv(d.U),cholinv(d.V))
BayesBase.vague(::Type{<:MatrixNormal}, dims::Tuple{Int,Int}) =
    MatrixNormal(zeros(Float64, dims), diagm(ones(dims[1])), diagm(ones(dims[2])))

getsupport(::MatrixNormal) = ProductDomain(VectorDomain{FullSpace},VectorDomain{FullSpace})

function Base.convert(::Type{MvNormalMeanCovariance}, d::MatrixNormal)
    covmat = cov(d)
    return MvNormalMeanCovariance(vec(mean(d)), Distributions.PDMats.PDMat(kron(covmat[1],covmat[2])))
end

BayesBase.default_prod_rule(::Type{MatrixNormal}, ::Type{MatrixNormal}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{MatrixNormal}, left::MatrixNormal, right::MatrixNormal) 
    error("The product of matrix Normals is not MatrixNormal")
end

function BayesBase.prod(
    ::PreserveTypeProd{ExponentialFamilyDistribution},
    left::ExponentialFamilyDistribution{T},
    right::ExponentialFamilyDistribution{T}
) where {T <: MatrixNormal}
    η_left = getnaturalparameters(left)
    η_right = getnaturalparameters(right)

    naturalparameters = η_left + η_right
    basemeasure = (x) -> oneunit(x)
    sufficientstatistics = getsufficientstatistics(left)
    logpartition = getlogpartition(left)
    supp = getsupport(left)
    attributes = ExponentialFamilyDistributionAttributes(basemeasure, sufficientstatistics, logpartition, supp)

    return ExponentialFamilyDistribution(
        Matrixvariate,
        naturalparameters,
        nothing,
        attributes
    )
end

function (::MeanToNatural{MatrixNormal})(tuple_of_θ::Tuple{Any,Any,Any})
    (M,U,V) = tuple_of_θ
    # @assert isposdef(U)
    # @assert isposdef(V)
    return (M,cholinv(U),cholinv(V))
end

function (::NaturalToMean{MatrixNormal})(tuple_of_η::Tuple{Any,Any,Any})
    (M,invU,invV) = tuple_of_η
    # @assert isposdef(invU)
    # @assert isposdef(invV)
    return (M,cholinv(invU),cholinv(invV))
end

# Natural parameterization

isproper(::NaturalParametersSpace, ::Type{MatrixNormal}, η, conditioner) = isnothing(conditioner) && !any(isinf, η) && !any(isnan, η)
isproper(::MeanParametersSpace, ::Type{MatrixNormal}, θ, conditioner) = isnothing(conditioner) && !any(isinf, θ) && !any(isnan, θ)

isbasemeasureconstant(::Type{MatrixNormal}) = ConstantBaseMeasure()
getbasemeasure(::Type{MatrixNormal}) = (x) -> oneunit(x)

getnaturalparameters(::MeanParametersSpace, ::Type{MatrixNormal}) = (θ) -> begin
    return (mean(θ),precision(θ)...)    
end

getsufficientstatistics(::Type{MatrixNormal}) = (vec, (X) -> vec(X)*vec(X)')

# Mean parametrization

function BayesBase.entropy(d::MatrixNormal)
    (M,U,V) = params(d)
    n,p = size(d)
    return n*p/2*log(2*π) + p/2*logdet(U) + n/2*logdet(V) + n*p/2
end

getlogpartition(::MeanParametersSpace, ::Type{MatrixNormal}) = (θ) -> begin
    (M,U,V) = params(θ)
    n,p = size(M)
    return n*p/2*log(2*π) + p/2*logdet(U) + n/2*logdet(V)
end

getgradlogpartition(::MeanParametersSpace, ::Type{MatrixNormal}) = (θ) -> begin
    (M,U,V) = params(θ)
    n,p = size(M)
    return (zeros(n,p), p/2*cholinv(U), n/2*cholinv(V))
end

getfisherinformation(::MeanParametersSpace, ::Type{MatrixNormal}) = (θ) -> begin
    (M,U,V) = params(θ)
    n,p = size(M)
    iU = cholinv(U)
    iV = cholinv(V)
    return (kron(iU,iV), n/2*kron(iU,iU), p/2*kron(iV,iV)) 
end