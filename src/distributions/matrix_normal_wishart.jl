export MatrixNormalWishart

import BayesBase
import RxInfer
using LinearAlgebra
using Distributions
using SpecialFunctions


struct MatrixNormalWishart{T, A <: AbstractArray{T}, S <: Real} <: ContinuousMatrixDistribution
 
    M::A # Mean matrix
    Λ::A # Row precision matrix
    Ω::A # Inverse scale matrix
    ν::S # Degrees of freedom

    function MatrixNormalWishart(M::A, Λ::A, Ω::A, ν::S) where {T, A <: AbstractArray{T}, S <: Real}
        
        Dx,Dy = size(M)
        if size(Λ, 1) != Dx; error("Number of rows of row precision matrix does not match mean matrix num rows."); end
        if size(Λ, 2) != Dx; error("Number of columns of row precision matrix does not match mean matrix num columns."); end
        if size(Ω, 1) != Dy; error("Number of rows of scale matrix does not match mean matrix num columns."); end

        return new{T,A,S}(M,Λ,Ω,ν)
    end
end

BayesBase.dim(d::MatrixNormalWishart) = size(d.M)
BayesBase.params(d::MatrixNormalWishart) = (d.M, d.Λ, d.Ω, d.ν)
BayesBase.mean(d::MatrixNormalWishart) = d.M
rprecision(d::MatrixNormalWishart) = d.Λ
rcov(d::MatrixNormalWishart) = cholinv(d.Λ)
scalem(d::MatrixNormalWishart) = d.Ω
degrees(d::MatrixNormalWishart) = d.ν

function BayesBase.pdf(dist::MatrixNormalWishart, Φ::Tuple)
    M, Λ, Ω, ν = RxInfer.params(dist)
    Dx,Dy = BayesBase.dim(dist)
    A,W = Φ
    return sqrt(det(Ω)^Dy*det(Λ)^ν/((2*π)^(Dx*Dy)*2^(ν*Dy)))*sqrt(det(W)^(ν+Dx-Dy-1))/multigamma(Dy,ν/2)*exp(-1/2*tr(W*((A-M)'*Λ*(A-M)+Ω)))
end

function BayesBase.logpdf(dist::MatrixNormalWishart, Φ::Tuple)
    M, Λ, Ω, ν = RxInfer.params(dist)
    Dx,Dy = BayesBase.dim(dist)
    A,W = Φ
    return Dy/2*logdet(Ω) + ν/2*logdet(Λ) - Dx*Dy/2*log(2π) - ν*Dy/2*log(2) + (ν+Dx-Dy-1)/2*logdet(W) - logmultigamma(Dy,ν/2) -1/2*tr(W*((A-M)'*Λ*(A-M)+Ω))
end

BayesBase.default_prod_rule(::Type{<:MatrixNormalWishart}, ::Type{<:MatrixNormalWishart}) = BayesBase.PreserveTypeProd(Distribution)

function BayesBase.prod(::BayesBase.PreserveTypeProd{Distribution}, left::MatrixNormalWishart, right::MatrixNormalWishart)
    
    Dx,Dy = BayesBase.dim(left)
    Ml, Λl, Ωl, νl = RxInfer.params(left)
    Mr, Λr, Ωr, νr = RxInfer.params(right)

    Λ = Λl + Λr
    M = inv(Λl + Λr)*(Λl*Ml + Λr*Mr)
    Ω = Ωl + Ωr + Ml'*Λl*Ml + Mr'*Λr*Mr - (Λl*Ml + Λr*Mr)'*inv(Λl + Λr)*(Λl*Ml + Λr*Mr)
    ν = νl + νr + Dx - Dy - 1

    return MatrixNormalWishart(M, Λ, Ω, ν)
end

function multigamma(p,a)
    result = π^(p*(p-1)/4)
    for j = 1:p 
        result *= gamma(a + (1-j)/2)
    end
    return result
end

function logmultigamma(p,a)
    result = p*(p-1)/4*log(π)
    for j = 1:p 
        result += loggamma(a + (1-j)/2)
    end
    return result
end