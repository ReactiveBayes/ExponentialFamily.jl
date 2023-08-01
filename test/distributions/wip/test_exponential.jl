module ExponentialTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff
using StableRNGs

import ExponentialFamily:
    mirrorlog, ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

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
        @test isproper(ExponentialFamilyDistribution(Exponential, [-5.0])) === true
        @test isproper(ExponentialFamilyDistribution(Exponential, [1.0])) === false
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Exponential(1)) ≈ -MathConstants.eulergamma
        @test mean(log, Exponential(10)) ≈ 1.7253694280925127
        @test mean(log, Exponential(0.1)) ≈ -2.8798007578955787
    end

    @testset "logpdf" begin
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

    @testset "logpartition" begin
        distribution = Exponential(5)
        @test logpartition(distribution) ≈ logpartition(convert(ExponentialFamilyDistribution, distribution))
        distribution = Exponential(10)
        @test logpartition(distribution) ≈ logpartition(convert(ExponentialFamilyDistribution, distribution))
        distribution = Exponential(0.1)
        @test logpartition(distribution) ≈ logpartition(convert(ExponentialFamilyDistribution, distribution))
        distribution = Exponential(1)
        @test logpartition(distribution) ≈ logpartition(convert(ExponentialFamilyDistribution, distribution))
        distribution = Exponential(1)
        @test logpartition(distribution) ≈ logpartition(convert(ExponentialFamilyDistribution, distribution))
    end

    @testset "getters" begin
        left = convert(ExponentialFamilyDistribution, Exponential(4))
        right = convert(ExponentialFamilyDistribution, Exponential(3))
        @test getnaturalparameters(prod(left, right)) ≈ [-0.5833333333333333]

        left = convert(ExponentialFamilyDistribution, Exponential(4))
        right = convert(ExponentialFamilyDistribution, Exponential(5))
        @test getnaturalparameters(prod(left, right)) ≈ [-0.45]

        left = convert(ExponentialFamilyDistribution, Exponential(1))
        right = convert(ExponentialFamilyDistribution, Exponential(1))
        @test getnaturalparameters(prod(left, right)) ≈ [-2]
    end

    @testset "convert" begin
        @test convert(ExponentialFamilyDistribution, Exponential(5)) ==
              ExponentialFamilyDistribution(Exponential, [-0.2])
        @test convert(ExponentialFamilyDistribution, Exponential(1e12)) ==
              ExponentialFamilyDistribution(Exponential, [-1e-12])
    end

    transformation(η) = -inv(η[1])

    @testset "fisher information" begin
        for θ in 1:20
            dist = Exponential(θ)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Exponential, η))
            autograd_inforamation = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test first(fisherinformation(ef)) ≈ first(autograd_inforamation(η))
            J = ForwardDiff.gradient(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ first(fisherinformation(ef))
        end
    end

    @testset "ExponentialFamilyDistribution mean,var" begin
        for θ in 1:20
            dist = Exponential(θ)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end
