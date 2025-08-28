export MatrixNormal

import BayesBase
import RxInfer
using LinearAlgebra
using Distributions
using SpecialFunctions


struct MatrixNormal{T, A <: AbstractArray{T}} <: ContinuousMatrixDistribution
 
    M::A # Mean matrix
    U::A # Row covariance matrix
    V::A # Column covariance matrix

    function MatrixNormal(M::A, U::A, V::A) where {T, A <: AbstractArray{T}}
        
        Dx,Dy = size(M)
        if size(U, 1) != Dx; error("Number of rows of row covariance matrix does not match mean matrix num rows."); end
        if size(U, 2) != Dx; error("Number of columns of row covariance matrix does not match mean matrix num columns."); end
        if size(V, 1) != Dy; error("Number of rows of column covariance matrix does not match mean matrix num columns."); end

        return new{T,A}(M,U,V)
    end
end

BayesBase.dim(d::MatrixNormal) = size(d.M)
BayesBase.params(d::MatrixNormal) = (d.M, d.U, d.V)
BayesBase.mean(d::MatrixNormal) = d.M
rowcov(d::MatrixNormal) = d.U
colcov(d::MatrixNormal) = d.V

function BayesBase.pdf(dist::MatrixNormal, X::Matrix)
    M, U, V = RxInfer.params(dist)
    Dx,Dy = BayesBase.dim(dist)
    return error("todo")
end

function BayesBase.logpdf(dist::MatrixNormal, X::Matrix)
    M, U, V = RxInfer.params(dist)
    Dx,Dy = BayesBase.dim(dist)
    return error("todo")
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