module LogNormalTest

using Test
using Random
using Distributions
using ExponentialFamily
import ExponentialFamily: KnownExponentialFamilyDistribution, basemeasure

@testset "LogNormal" begin
    @testset "Constructors" begin
        @test LogNormal() == LogNormal(0.0, 1.0)
        @test typeof(vague(LogNormal)) <: LogNormal
        @test vague(LogNormal) == LogNormal(1, 1e12)
    end

    @testset "prod" begin
        @test prod(ClosedProd(), LogNormal(1.0, 1.0), LogNormal(1.0, 1.0)) == LogNormal(0.5, 0.5)
        @test prod(ClosedProd(), LogNormal(2.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.5, 0.5)
        @test prod(ClosedProd(), LogNormal(1.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.0, 0.5)
        @test prod(ClosedProd(), LogNormal(1.0, 2.0), LogNormal(1.0, 2.0)) == LogNormal(0.0, 1.0)
        @test prod(ClosedProd(), LogNormal(2.0, 2.0), LogNormal(2.0, 2.0)) == LogNormal(1.0, 1.0)
    end

    @testset "Base operations" begin
        @test prod(KnownExponentialFamilyDistribution(LogNormal, [1, 1]), KnownExponentialFamilyDistribution(LogNormal, [1, 1])) ==
              KnownExponentialFamilyDistribution(LogNormal, [2, 2])
        @test prod(KnownExponentialFamilyDistribution(LogNormal, [2, 3]), KnownExponentialFamilyDistribution(LogNormal, [-1, -2])) ==       
              KnownExponentialFamilyDistribution(LogNormal, [1, 1])
    end

    @testset "Natural parameterization" begin
        @test Distributions.logpdf(LogNormal(10, 4.0), 1.0) ≈
              Distributions.logpdf(convert(KnownExponentialFamilyDistribution, LogNormal(10, 4.0)), 1.0)
        @test Distributions.logpdf(LogNormal(5, 2.0), 1.0) ≈
              Distributions.logpdf(convert(KnownExponentialFamilyDistribution, LogNormal(5, 2.0)), 1.0)
    end

    @testset "LogNormalKnownExponentialFamilyDistribution" begin
        @test convert(Distribution, KnownExponentialFamilyDistribution(LogNormal, [1, -1])) ≈ LogNormal(1, 0.5)
        @test Distributions.logpdf(KnownExponentialFamilyDistribution(LogNormal, [2, -2]), 10) ≈
              Distributions.logpdf(LogNormal(0.75, 0.25), 10)
        @test isproper(KnownExponentialFamilyDistribution(LogNormal, [1, -2])) === true
        @test isproper(KnownExponentialFamilyDistribution(LogNormal, [2, 1])) === false

        @test convert(KnownExponentialFamilyDistribution, LogNormal(1.0, 2.0)) == KnownExponentialFamilyDistribution(LogNormal, [-0.5, -0.25])
    end

    @testset "logpartition" begin
        @test logpartition(KnownExponentialFamilyDistribution(LogNormal, [1, -2])) ≈ 0.5
        @test logpartition(KnownExponentialFamilyDistribution(LogNormal, [1, -1])) ≈ 1.0
    end

    @testset "basemeasure" begin
        @test basemeasure(LogNormal(1, 1 / 2pi), rand(1)) == 1.0
        @test basemeasure(KnownExponentialFamilyDistribution(LogNormal, [1.0, -pi]), rand(1)) == 1.0
    end
end
end
