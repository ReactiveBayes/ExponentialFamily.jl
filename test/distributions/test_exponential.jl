module ExponentialTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: mirrorlog, ExponentialFamilyDistribution, getnaturalparameters, basemeasure

@testset "Exponential" begin

    # Beta comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(Exponential)

        @test typeof(d) <: Exponential
        @test mean(d) === 1e12
        @test params(d) === (1e12,)
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Exponential(5), Exponential(4)) ≈ Exponential(1 / 0.45)
        @test prod(ProdAnalytical(), Exponential(1), Exponential(1)) ≈ Exponential(1 / 2)
        @test prod(ProdAnalytical(), Exponential(0.1), Exponential(0.1)) ≈ Exponential(0.05)
    end

    @testset "isproper" begin
        @test isproper(ExponentialFamilyDistribution(Exponential, [-5.0])) === true
        @test isproper(ExponentialFamilyDistribution(Exponential, [1.0])) === false
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Exponential(1)) ≈ -MathConstants.eulergamma
        @test mean(log, Exponential(10)) ≈ 1.7253694280925127
        @test mean(log, Exponential(0.1)) ≈ -2.8798007578955787
    end

    @testset "Constructor(::ExponentialExponentialFamilyDistribution)" begin
        @test convert(ExponentialFamilyDistribution, Exponential(5)) == ExponentialFamilyDistribution(Exponential, [-0.2])
        @test convert(ExponentialFamilyDistribution, Exponential(1e12)) == ExponentialFamilyDistribution(Exponential, [-1e-12])
        @test basemeasure(Exponential(5), rand()) == 1.0
    end

    @testset "logpdf(::ExponentialExponentialFamilyDistribution)" begin
        distribution = Exponential(5)
        @test logpdf(distribution, 1) ≈ logpdf(convert(ExponentialFamilyDistribution, distribution), 1)
        distribution = Exponential(10)
        @test logpdf(distribution, 1) ≈ logpdf(convert(ExponentialFamilyDistribution, distribution), 1)
        distribution = Exponential(0.1)
        @test logpdf(distribution, 0) ≈ logpdf(convert(ExponentialFamilyDistribution, distribution), 0)
        distribution = Exponential(1)
        @test logpdf(distribution, 1) ≈ logpdf(convert(ExponentialFamilyDistribution, distribution), 1)
        distribution = Exponential(1)
        @test logpdf(distribution, 2) ≈ logpdf(convert(ExponentialFamilyDistribution, distribution), 2)
    end

    @testset "lognormalizer(::ExponentialExponentialFamilyDistribution)" begin
        distribution = Exponential(5)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(ExponentialFamilyDistribution, distribution))
        distribution = Exponential(10)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(ExponentialFamilyDistribution, distribution))
        distribution = Exponential(0.1)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(ExponentialFamilyDistribution, distribution))
        distribution = Exponential(1)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(ExponentialFamilyDistribution, distribution))
        distribution = Exponential(1)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(ExponentialFamilyDistribution, distribution))
    end

    @testset ":+(::ExponentialExponentialFamilyDistribution, ::ExponentialExponentialFamilyDistribution)" begin
        left = convert(ExponentialFamilyDistribution, Exponential(4))
        right = convert(ExponentialFamilyDistribution, Exponential(3))
        @test getnaturalparameters(left + right) ≈ [-0.5833333333333333]

        left = convert(ExponentialFamilyDistribution, Exponential(4))
        right = convert(ExponentialFamilyDistribution, Exponential(5))
        @test getnaturalparameters(left + right) ≈ [-0.45]

        left = convert(ExponentialFamilyDistribution, Exponential(1))
        right = convert(ExponentialFamilyDistribution, Exponential(1))
        @test getnaturalparameters(left + right) ≈ [-2]
    end

    @testset ":-(::ExponentialExponentialFamilyDistribution, ::ExponentialExponentialFamilyDistribution)" begin
        left = convert(ExponentialFamilyDistribution, Exponential(1))
        right = convert(ExponentialFamilyDistribution, Exponential(4))
        @test getnaturalparameters(left - right) ≈ [-0.75]

        left = convert(ExponentialFamilyDistribution, Exponential(4))
        right = convert(ExponentialFamilyDistribution, Exponential(5))
        @test getnaturalparameters(left - right) ≈ [-0.05]

        left = convert(ExponentialFamilyDistribution, Exponential(1))
        right = convert(ExponentialFamilyDistribution, Exponential(1))
        @test getnaturalparameters(left - right) ≈ [0]
    end
end

end
