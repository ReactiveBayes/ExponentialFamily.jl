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
            dist = MvNormalMeanCovariance([m, m], [sigma 0; 0 sigma])
            ef = convert(KnownExponentialFamilyDistribution, dist)
            vec = [getnaturalparameters(ef)[1]..., getnaturalparameters(ef)[2]...]
            autograd_hessian = ForwardDiff.hessian(x -> reconstructed_logpartition(ef, x), vec)
            @info "test started"
            # display(autograd_hessian)
            # display(cov(dist))
            # display(getnaturalparameters(ef)[2])
            # display(fisherinformation(dist))
            display(fisherinformation(ef))
        end
    end
end

end