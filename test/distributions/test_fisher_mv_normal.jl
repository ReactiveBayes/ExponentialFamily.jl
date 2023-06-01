module NormalTest

using Test
using ExponentialFamily
using LinearAlgebra
using Distributions
using ForwardDiff
using Random
using StableRNGs

import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation, reconstructargument!

function reconstructed_logpartition(ef::KnownExponentialFamilyDistribution{T}, ηvec) where {T}
    natural_params = getnaturalparameters(ef)
    mean_size = length(natural_params[1])
    @views wmean = ηvec[1:mean_size]
    @views matrix = reshape(ηvec[(mean_size+1):end], mean_size, mean_size)
    ef = KnownExponentialFamilyDistribution(T, [wmean, matrix])
    return logpartition(ef)
end

@testset "Normal" begin
    @testset "MultivariateNormalKnownExponentialFamilyDistribution" begin
        for (m, sigma) in zip(1:10, 1:10)
            cov = [sigma 0; 0 sigma]
            dist = convert(KnownExponentialFamilyDistribution, MvNormalMeanCovariance([0, 0], cov))
            vec = [getnaturalparameters(dist)[1]..., getnaturalparameters(dist)[2]...]
            autograd_hessian = ForwardDiff.hessian(x -> reconstructed_logpartition(dist, x), vec)
            @info (dist, autograd_hessian)
            @info autograd_hessian
            display(autograd_hessian)
            display(cov)
            display(2*kron(cov, cov))
            display(getnaturalparameters(dist)[2])
            @info fisherinformation(dist)
        end
    end
end

end