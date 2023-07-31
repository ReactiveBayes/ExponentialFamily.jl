module LogNormalTest

using Test
using Random
using Distributions
using StableRNGs
using Zygote
using ForwardDiff
using ExponentialFamily
import ExponentialFamily: ExponentialFamilyDistribution, basemeasure, fisherinformation, getnaturalparameters

@testset "LogNormal" begin
    @testset "constructors" begin
        @test LogNormal() == LogNormal(0.0, 1.0)
        @test typeof(vague(LogNormal)) <: LogNormal
        @test vague(LogNormal) == LogNormal(1, 1e12)
    end

    @testset "prod" begin
        @test prod(ClosedProd(), LogNormal(1.0, 1.0), LogNormal(1.0, 1.0)) == LogNormal(0.5, sqrt(1 / 2))
        @test prod(ClosedProd(), LogNormal(2.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.5, sqrt(1 / 2))
        @test prod(ClosedProd(), LogNormal(1.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.0, sqrt(1 / 2))
        @test prod(ClosedProd(), LogNormal(1.0, 2.0), LogNormal(1.0, 2.0)) == LogNormal(-1.0, sqrt(2))
        @test prod(ClosedProd(), LogNormal(2.0, 2.0), LogNormal(2.0, 2.0)) == LogNormal(0.0, sqrt(2))
    end

    @testset "Base operations" begin
        @test prod(
            ExponentialFamilyDistribution(LogNormal, [1, 1]),
            ExponentialFamilyDistribution(LogNormal, [1, 1])
        ) ==
              ExponentialFamilyDistribution(LogNormal, [2, 2])
        @test prod(
            ExponentialFamilyDistribution(LogNormal, [2, 3]),
            ExponentialFamilyDistribution(LogNormal, [-1, -2])
        ) ==
              ExponentialFamilyDistribution(LogNormal, [1, 1])
    end

    @testset "logpdf" begin
        @test Distributions.logpdf(LogNormal(10, 4.0), 1.0) ≈
              Distributions.logpdf(convert(ExponentialFamilyDistribution, LogNormal(10, 4.0)), 1.0)
        @test Distributions.logpdf(LogNormal(5, 2.0), 1.0) ≈
              Distributions.logpdf(convert(ExponentialFamilyDistribution, LogNormal(5, 2.0)), 1.0)
    end

    @testset "convert" begin
        @test convert(Distribution, ExponentialFamilyDistribution(LogNormal, [1, -1])) ≈
              LogNormal(1 / 2, sqrt(0.5))
        @test Distributions.logpdf(ExponentialFamilyDistribution(LogNormal, [2, -2]), 10) ≈
              Distributions.logpdf(LogNormal(0.5, sqrt(0.25)), 10)

        @test convert(ExponentialFamilyDistribution, LogNormal(1.0, sqrt(2.0))) ≈
              ExponentialFamilyDistribution(LogNormal, [0.5, -0.25])
    end

    @testset "isproper" begin
        @test isproper(ExponentialFamilyDistribution(LogNormal, [1, -2])) === true
        @test isproper(ExponentialFamilyDistribution(LogNormal, [2, 1])) === false
    end

    @testset "logpartition" begin
        @test logpartition(ExponentialFamilyDistribution(LogNormal, [1, -2])) ≈ 1 / 8 - log(4) / 2
        @test logpartition(ExponentialFamilyDistribution(LogNormal, [1, -1])) ≈ 1 / 4 - log(2) / 2
    end

    @testset "basemeasure" begin
        point = rand()
        @test basemeasure(ExponentialFamilyDistribution(LogNormal, [1.0, -pi]), point) ==
              1 / (sqrt(2 * pi) * point)
        @test basemeasure(ExponentialFamilyDistribution(LogNormal, [1.0, -1]), point) == 1 / (sqrt(2 * pi) * point)
    end

    @testset "fisher information" begin
        transformation(η) = [-(η[1]) / (2 * η[2]), sqrt(-1 / (2 * η[2]))]
        for λ in 1:10, σ in 1:10
            dist = LogNormal(λ, σ)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(LogNormal, η))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-8
            J = ForwardDiff.jacobian(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) atol = 1e-8
        end
    end

    @testset "ExponentialFamilyDistribution mean,var" begin
        for λ in 0.1:0.1:1.0, σ in 0.1:0.1:0.5
            dist = LogNormal(λ, σ)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end
