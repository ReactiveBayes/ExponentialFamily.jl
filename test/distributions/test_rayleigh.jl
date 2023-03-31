module RayleighTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: mirrorlog, KnownExponentialFamilyDistribution, logpartition, basemeasure

@testset "Rayleigh" begin
    @testset "vague" begin
        d = vague(Rayleigh)

        @test typeof(d) <: Rayleigh
        @test mean(d) ≈ d.σ * √(π / 2)
        @test params(d) === (1e12,)
    end

    @testset "prod" begin
        @test prod(ClosedProd(), Rayleigh(3.0), Rayleigh(2.0)) ≈ Rayleigh(sqrt(36 / 13))
        @test prod(ClosedProd(), Rayleigh(7.0), Rayleigh(1.0)) ≈ Rayleigh(sqrt(49 / 50))
        @test prod(ClosedProd(), Rayleigh(1.0), Rayleigh(2.0)) ≈ Rayleigh(sqrt(4 / 5))
    end

    @testset "RayleighKnownExponentialFamilyDistribution" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(Distribution, KnownExponentialFamilyDistribution(Rayleigh, [-i])) ==
                      Rayleigh(sqrt(1 / 2i))

                @test convert(KnownExponentialFamilyDistribution, Rayleigh(i)) ==
                      KnownExponentialFamilyDistribution(Rayleigh, [-1 / (2i^2)])
            end
        end

        @testset "logpartition" begin
            @test logpartition(KnownExponentialFamilyDistribution(Rayleigh, -1.0)) ≈ log(2)
            @test logpartition(KnownExponentialFamilyDistribution(Rayleigh, -2.0)) ≈ log(4)
        end

        @testset "logpdf" begin
            for i in 1:10
                @test logpdf(KnownExponentialFamilyDistribution(Rayleigh, [-i]), 0.01) ≈
                      logpdf(Rayleigh(sqrt(1 / 2i)), 0.01)
                @test logpdf(KnownExponentialFamilyDistribution(Rayleigh, [-i]), 0.5) ≈
                      logpdf(Rayleigh(sqrt(1 / 2i)), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(Rayleigh, [-i])) === true
                @test isproper(KnownExponentialFamilyDistribution(Rayleigh, [i])) === false
            end
        end

        @testset "basemeasure" begin
            for (i) in (1:10)
                @test basemeasure(KnownExponentialFamilyDistribution(Rayleigh, [-i]), i^2) == i^2
                @test basemeasure(Rayleigh(i), i / 2) == i / 2
            end
        end
    end
end

end
