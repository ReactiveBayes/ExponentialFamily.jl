module ExponentialTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: mirrorlog

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

    @testset "mean(::typeof(log))" begin
        @test mean(log, Exponential(1)) ≈ -MathConstants.eulergamma
        @test mean(log, Exponential(10)) ≈ 1.7253694280925127
        @test mean(log, Exponential(0.1)) ≈ -2.8798007578955787
    end

    @testset "Constructor(::ExponentialNaturalParameters)" begin
        @test naturalparams(Exponential(5)) == ExponentialNaturalParameters(-0.2)
        @test naturalparams(Exponential(1e12)) == ExponentialNaturalParameters(-1e-12)
    end

    @testset "logpdf(::ExponentialNaturalParameters)" begin
        distribution = Exponential(5)
        @test logpdf(distribution, 1) ≈ logpdf(naturalparams(distribution), 1)
        distribution = Exponential(10)
        @test logpdf(distribution, 1) ≈ logpdf(naturalparams(distribution), 1)
        distribution = Exponential(0.1)
        @test logpdf(distribution, 0) ≈ logpdf(naturalparams(distribution), 0)
        distribution = Exponential(1)
        @test logpdf(distribution, 1) ≈ logpdf(naturalparams(distribution), 1)
        distribution = Exponential(1)
        @test logpdf(distribution, 2) ≈ logpdf(naturalparams(distribution), 2)
    end

    @testset "lognormalizer(::ExponentialNaturalParameters)" begin
        distribution = Exponential(5)
        @test lognormalizer(distribution) ≈ lognormalizer(naturalparams(distribution))
        distribution = Exponential(10)
        @test lognormalizer(distribution) ≈ lognormalizer(naturalparams(distribution))
        distribution = Exponential(0.1)
        @test lognormalizer(distribution) ≈ lognormalizer(naturalparams(distribution))
        distribution = Exponential(1)
        @test lognormalizer(distribution) ≈ lognormalizer(naturalparams(distribution))
        distribution = Exponential(1)
        @test lognormalizer(distribution) ≈ lognormalizer(naturalparams(distribution))
    end

    @testset ":+(::ExponentialNaturalParameters, ::ExponentialNaturalParameters)" begin
        left = naturalparams(Exponential(4))
        right = naturalparams(Exponential(3))
        @test (left + right).minus_rate ≈ -0.5833333333333333

        left = naturalparams(Exponential(4))
        right = naturalparams(Exponential(5))
        @test (left + right).minus_rate ≈ -0.45

        left = naturalparams(Exponential(1))
        right = naturalparams(Exponential(1))
        @test (left + right).minus_rate ≈ -2
    end

    @testset ":-(::ExponentialNaturalParameters, ::ExponentialNaturalParameters)" begin
        left = naturalparams(Exponential(1))
        right = naturalparams(Exponential(4))
        @test (left - right).minus_rate ≈ -0.75

        left = naturalparams(Exponential(4))
        right = naturalparams(Exponential(5))
        @test (left - right).minus_rate ≈ -0.05

        left = naturalparams(Exponential(1))
        right = naturalparams(Exponential(1))
        @test (left - right).minus_rate ≈ 0
    end
end

end
