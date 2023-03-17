module ExponentialTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: mirrorlog, NaturalParameters, get_params, basemeasure

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
        @test isproper(NaturalParameters(Exponential, [-5.0])) === true
        @test isproper(NaturalParameters(Exponential, [1.0])) === false
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Exponential(1)) ≈ -MathConstants.eulergamma
        @test mean(log, Exponential(10)) ≈ 1.7253694280925127
        @test mean(log, Exponential(0.1)) ≈ -2.8798007578955787
    end

    @testset "Constructor(::ExponentialNaturalParameters)" begin
        @test convert(NaturalParameters, Exponential(5)) == NaturalParameters(Exponential, [-0.2])
        @test convert(NaturalParameters, Exponential(1e12)) == NaturalParameters(Exponential, [-1e-12])
        @test basemeasure(Exponential(5), rand()) == 1.0
    end

    @testset "logpdf(::ExponentialNaturalParameters)" begin
        distribution = Exponential(5)
        @test logpdf(distribution, 1) ≈ logpdf(convert(NaturalParameters, distribution), 1)
        distribution = Exponential(10)
        @test logpdf(distribution, 1) ≈ logpdf(convert(NaturalParameters, distribution), 1)
        distribution = Exponential(0.1)
        @test logpdf(distribution, 0) ≈ logpdf(convert(NaturalParameters, distribution), 0)
        distribution = Exponential(1)
        @test logpdf(distribution, 1) ≈ logpdf(convert(NaturalParameters, distribution), 1)
        distribution = Exponential(1)
        @test logpdf(distribution, 2) ≈ logpdf(convert(NaturalParameters, distribution), 2)
    end

    @testset "lognormalizer(::ExponentialNaturalParameters)" begin
        distribution = Exponential(5)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(NaturalParameters, distribution))
        distribution = Exponential(10)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(NaturalParameters, distribution))
        distribution = Exponential(0.1)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(NaturalParameters, distribution))
        distribution = Exponential(1)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(NaturalParameters, distribution))
        distribution = Exponential(1)
        @test lognormalizer(distribution) ≈ lognormalizer(convert(NaturalParameters, distribution))
    end

    @testset ":+(::ExponentialNaturalParameters, ::ExponentialNaturalParameters)" begin
        left = convert(NaturalParameters, Exponential(4))
        right = convert(NaturalParameters, Exponential(3))
        @test get_params(left + right) ≈ [-0.5833333333333333]

        left = convert(NaturalParameters, Exponential(4))
        right = convert(NaturalParameters, Exponential(5))
        @test get_params(left + right) ≈ [-0.45]

        left = convert(NaturalParameters, Exponential(1))
        right = convert(NaturalParameters, Exponential(1))
        @test get_params(left + right) ≈ [-2]
    end

    @testset ":-(::ExponentialNaturalParameters, ::ExponentialNaturalParameters)" begin
        left = convert(NaturalParameters, Exponential(1))
        right = convert(NaturalParameters, Exponential(4))
        @test get_params(left - right) ≈ [-0.75]

        left = convert(NaturalParameters, Exponential(4))
        right = convert(NaturalParameters, Exponential(5))
        @test get_params(left - right) ≈ [-0.05]

        left = convert(NaturalParameters, Exponential(1))
        right = convert(NaturalParameters, Exponential(1))
        @test get_params(left - right) ≈ [0]
    end
end

end
