module VonMisesTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, logpartition, basemeasure

@testset "Weibull" begin

    # Weibull comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "convert" begin
        for λ in 0.5:0.5:10, k in 0.5:0.5:10
            @test convert(KnownExponentialFamilyDistribution, Weibull(k, λ)) ≈
                  KnownExponentialFamilyDistribution(Weibull, [-(1 / λ)^(k)], k)
        end
    end

    @testset "logpartition" begin
        @test logpartition(KnownExponentialFamilyDistribution(Weibull, [-1], 1)) ≈ 0
    end
end

end
