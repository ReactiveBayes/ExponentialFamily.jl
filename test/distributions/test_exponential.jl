module ExponentialTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff
using StableRNGs

import ExponentialFamily:
    mirrorlog, KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

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
        @test isproper(KnownExponentialFamilyDistribution(Exponential, -5.0)) === true
        @test isproper(KnownExponentialFamilyDistribution(Exponential, 1.0)) === false
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
        @test getnaturalparameters(prod(left, right)) ≈ -0.5833333333333333

        left = convert(KnownExponentialFamilyDistribution, Exponential(4))
        right = convert(KnownExponentialFamilyDistribution, Exponential(5))
        @test getnaturalparameters(prod(left, right)) ≈ -0.45

        left = convert(KnownExponentialFamilyDistribution, Exponential(1))
        right = convert(KnownExponentialFamilyDistribution, Exponential(1))
        @test getnaturalparameters(prod(left, right)) ≈ -2
    end

    @testset "convert" begin
        @test convert(KnownExponentialFamilyDistribution, Exponential(5)) ==
              KnownExponentialFamilyDistribution(Exponential, -0.2)
        @test convert(KnownExponentialFamilyDistribution, Exponential(1e12)) ==
              KnownExponentialFamilyDistribution(Exponential, -1e-12)
    end

    transformation(η) = -inv(η)

    @testset "fisher information" begin
        for θ in 1:20
            dist = Exponential(θ)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(Exponential, η))
            df = (η) -> ForwardDiff.derivative(f_logpartition, η)
            autograd_inforamation = (η) -> ForwardDiff.derivative(df, η)
            @test fisherinformation(ef) ≈ autograd_inforamation(η)
            J = ForwardDiff.derivative(transformation, η)
            @test fisherinformation(dist) * J^2 ≈ fisherinformation(ef)
        end
    end

    @testset "KnownExponentialFamilyDistribution mean,var" begin
        for θ in 1:20
            dist = Exponential(θ)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end
