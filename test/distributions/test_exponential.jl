module ExponentialTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff

import ExponentialFamily:
    mirrorlog, KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisher_information

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
        @test prod(ClosedProd(), Exponential(5), Exponential(4)) ≈ Exponential(1 / 0.45)
        @test prod(ClosedProd(), Exponential(1), Exponential(1)) ≈ Exponential(1 / 2)
        @test prod(ClosedProd(), Exponential(0.1), Exponential(0.1)) ≈ Exponential(0.05)
    end

    @testset "isproper" begin
        @test isproper(KnownExponentialFamilyDistribution(Exponential, [-5.0])) === true
        @test isproper(KnownExponentialFamilyDistribution(Exponential, [1.0])) === false
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Exponential(1)) ≈ -MathConstants.eulergamma
        @test mean(log, Exponential(10)) ≈ 1.7253694280925127
        @test mean(log, Exponential(0.1)) ≈ -2.8798007578955787
    end

    @testset "logpdf" begin
        distribution = Exponential(5)
        @test logpdf(distribution, 1) ≈ logpdf(convert(KnownExponentialFamilyDistribution, distribution), 1)
        distribution = Exponential(10)
        @test logpdf(distribution, 1) ≈ logpdf(convert(KnownExponentialFamilyDistribution, distribution), 1)
        distribution = Exponential(0.1)
        @test logpdf(distribution, 0) ≈ logpdf(convert(KnownExponentialFamilyDistribution, distribution), 0)
        distribution = Exponential(1)
        @test logpdf(distribution, 1) ≈ logpdf(convert(KnownExponentialFamilyDistribution, distribution), 1)
        distribution = Exponential(1)
        @test logpdf(distribution, 2) ≈ logpdf(convert(KnownExponentialFamilyDistribution, distribution), 2)
    end

    @testset "logpartition" begin
        distribution = Exponential(5)
        @test logpartition(distribution) ≈ logpartition(convert(KnownExponentialFamilyDistribution, distribution))
        distribution = Exponential(10)
        @test logpartition(distribution) ≈ logpartition(convert(KnownExponentialFamilyDistribution, distribution))
        distribution = Exponential(0.1)
        @test logpartition(distribution) ≈ logpartition(convert(KnownExponentialFamilyDistribution, distribution))
        distribution = Exponential(1)
        @test logpartition(distribution) ≈ logpartition(convert(KnownExponentialFamilyDistribution, distribution))
        distribution = Exponential(1)
        @test logpartition(distribution) ≈ logpartition(convert(KnownExponentialFamilyDistribution, distribution))
    end

    @testset "getters" begin
        left = convert(KnownExponentialFamilyDistribution, Exponential(4))
        right = convert(KnownExponentialFamilyDistribution, Exponential(3))
        @test getnaturalparameters(prod(left, right)) ≈ [-0.5833333333333333]

        left = convert(KnownExponentialFamilyDistribution, Exponential(4))
        right = convert(KnownExponentialFamilyDistribution, Exponential(5))
        @test getnaturalparameters(prod(left, right)) ≈ [-0.45]

        left = convert(KnownExponentialFamilyDistribution, Exponential(1))
        right = convert(KnownExponentialFamilyDistribution, Exponential(1))
        @test getnaturalparameters(prod(left, right)) ≈ [-2]
    end

    @testset "convert" begin
        @test convert(KnownExponentialFamilyDistribution, Exponential(5)) ==
              KnownExponentialFamilyDistribution(Exponential, [-0.2])
        @test convert(KnownExponentialFamilyDistribution, Exponential(1e12)) ==
              KnownExponentialFamilyDistribution(Exponential, [-1e-12])
        @test basemeasure(Exponential(5), rand()) == 1.0
    end

    @testset "fisher information" begin
        θ = 5.0
        dist = Exponential(θ)
        ef = convert(KnownExponentialFamilyDistribution, dist)
        η = getnaturalparameters(ef)

        @test fisher_information(dist) ≈ 1 / θ^2 atol = 1e-8

        samples = rand(Exponential(θ), 10000)
        hessian = (x) -> -ForwardDiff.hessian((params) -> mean(logpdf.(Exponential(params[1]), samples)), x)
        fisher_information(Exponential(θ)) ≈ first(hessian([θ]))

        f_logpartion = (η) -> logpartition(KnownExponentialFamilyDistribution(Exponential, η))
        autograd_inforamation = (η) -> ForwardDiff.hessian(f_logpartion, η)
        @test fisher_information(ef) ≈ first(autograd_inforamation(η))
    end
end

end
