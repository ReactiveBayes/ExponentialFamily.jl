module LogNormalTest

using Test
using Random
using Distributions
using ExponentialFamily
import ExponentialFamily: NaturalParameters, basemeasure

@testset "LogNormal" begin
    @testset "Constructors" begin
        @test LogNormal() == LogNormal(0.0, 1.0)
        @test typeof(vague(LogNormal)) <: LogNormal
        @test vague(LogNormal) == LogNormal(1, 1e12)
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), LogNormal(1.0, 1.0), LogNormal(1.0, 1.0)) == LogNormal(0.5, 0.5)
        @test prod(ProdAnalytical(), LogNormal(2.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.5, 0.5)
        @test prod(ProdAnalytical(), LogNormal(1.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.0, 0.5)
        @test prod(ProdAnalytical(), LogNormal(1.0, 2.0), LogNormal(1.0, 2.0)) == LogNormal(0.0, 1.0)
        @test prod(ProdAnalytical(), LogNormal(2.0, 2.0), LogNormal(2.0, 2.0)) == LogNormal(1.0, 1.0)
    end

    @testset "Base operations" begin
        @test NaturalParameters(LogNormal, [1.0, 2.0]) - NaturalParameters(LogNormal, [2.0, 3.0]) ==
              NaturalParameters(LogNormal, [-1.0, -1.0])
        @test NaturalParameters(LogNormal, [4, 2.0]) + NaturalParameters(LogNormal, [2, 3.0]) ==
              NaturalParameters(LogNormal, [6, 5.0])
    end

    @testset "Natural parameterization" begin
        @test Distributions.logpdf(LogNormal(10, 4.0), 1.0) ≈
              Distributions.logpdf(convert(NaturalParameters, LogNormal(10, 4.0)), 1.0)
        @test Distributions.logpdf(LogNormal(5, 2.0), 1.0) ≈
              Distributions.logpdf(convert(NaturalParameters, LogNormal(5, 2.0)), 1.0)
    end

    @testset "NaturalParameters" begin
        @test convert(Distribution, NaturalParameters(LogNormal, [1, -1])) ≈ LogNormal(1, 0.5)
        @test Distributions.logpdf(NaturalParameters(LogNormal, [2, -2]), 10) ≈
              Distributions.logpdf(LogNormal(0.75, 0.25), 10)
        @test isproper(NaturalParameters(LogNormal, [1, -2])) === true
        @test isproper(NaturalParameters(LogNormal, [2, 1])) === false

        @test convert(NaturalParameters, LogNormal(1.0, 2.0)) == NaturalParameters(LogNormal, [-0.5, -0.25])
    end

    @testset "lognormalizer" begin
        @test lognormalizer(NaturalParameters(LogNormal, [1, -2])) ≈ 0.5
        @test lognormalizer(NaturalParameters(LogNormal, [1, -1])) ≈ 1.0
    end

    @testset "basemeasure" begin
        @test basemeasure(LogNormal(1, 1 / 2pi), rand(1)) == 1.0
        @test basemeasure(NaturalParameters(LogNormal, [1.0, -pi]), rand(1)) == 1.0
    end
end
end
