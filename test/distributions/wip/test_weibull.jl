module WeibullTest

using Test
using ExponentialFamily
using Distributions
using Random
using HCubature
using SpecialFunctions
using ForwardDiff
using StableRNGs

import ExponentialFamily: ExponentialFamilyDistribution, ExponentialFamilyDistribution, getnaturalparameters,
    getsufficientstatistics, getlogpartition, getbasemeasure, fisherinformation, getconditioner
import ExponentialFamily: basemeasure, isproper
import StatsFuns: xlogy

@testset "Weibull" begin

    # Weibull comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "convert" begin
        for λ in 0.5:0.5:10, k in 0.5:0.5:10
            @test convert(ExponentialFamilyDistribution, Weibull(k, λ)) ≈
                  ExponentialFamilyDistribution(Weibull, [-(1 / λ)^(k)], k)
        end
    end

    @testset "prod (same k)" begin
        for η in -10:0.5:-0.5, k in 1.0:0.5:10, x in 0.5:0.5:10
            ef_left = convert(Distribution, ExponentialFamilyDistribution(Weibull, [η], k))
            ef_right = convert(Distribution, ExponentialFamilyDistribution(Weibull, [-η^2], k))
            res = prod(ClosedProd(), ef_left, ef_right)
            @test getbasemeasure(res)(x) == x^(2 * (k - 1))
            @test getsufficientstatistics(res)(x) == [x^k]
            @test getlogpartition(res)(η - η^2) ==
                  log(abs(η - η^2)^(1 / k)) + loggamma(2 - 1 / k) - 2 * log(abs(η - η^2)) - log(k)
            @test getnaturalparameters(res) ≈ [η - η^2]
            @test first(hquadrature(x -> pdf(res, tan(x * pi / 2)) * (pi / 2) * (1 / cos(pi * x / 2))^2, 0.0, 1.0)) ≈
                  1.0
        end
    end

    @testset "prod (different k)" begin
        for η in -12:4:-0.5, k in 1.0:4:10, x in 0.5:4:10
            ef_left = convert(Distribution, ExponentialFamilyDistribution(Weibull, [η], k * 2))
            ef_right = convert(Distribution, ExponentialFamilyDistribution(Weibull, [-η^2], k))
            res = prod(ClosedProd(), ef_left, ef_right)
            @test getbasemeasure(res)(x) == x^(k + k * 2 - 2)
            @test getsufficientstatistics(res)(x) == [x^(2 * k), x^k]
            @test getnaturalparameters(res) ≈ [η, -η^2]
            @test first(hquadrature(x -> pdf(res, tan(x * pi / 2)) * (pi / 2) * (1 / cos(pi * x / 2))^2, 0.0, 1.0)) ≈
                  1.0
        end
    end

    @testset "logpartition" begin
        @test logpartition(ExponentialFamilyDistribution(Weibull, [-1], 1)) ≈ 0
    end

    @testset "isproper" begin
        for η in -10:0.5:10, k in 0.5:0.5:10
            @test isproper(ExponentialFamilyDistribution(Weibull, [η], k)) == (η < 0)
        end
    end

    @testset "basemeasure" begin
        for η in -10:0.5:-0.5, k in 0.5:0.5:10, x in 0.5:0.5:10
            @test basemeasure(ExponentialFamilyDistribution(Weibull, [η], k), x) ≈ x^(k - 1)
        end
    end

    @testset "isproper" begin
        for η in -12:4:-0.5, k in 1.0:4:10
            ef_proper = ExponentialFamilyDistribution(Weibull, [η], k)
            ef_improper = ExponentialFamilyDistribution(Weibull, [-η], k)
            @test isproper(ef_proper) == true
            @test isproper(ef_improper) == false
        end
    end

    @testset "fisher information" begin
        for (λ, k) in Iterators.product(1:5, 1:5)
            dist = Weibull(λ, k)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            transformation(η) = [getconditioner(ef), (-1 / η[1])^(1 / getconditioner(ef))]
            J = ForwardDiff.jacobian(transformation, η)
            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Weibull, η, k))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test first(fisherinformation(ef)) ≈ first(autograd_information(η)) atol = 1e-8
            @test first(J' * fisherinformation(dist) * J) ≈ first(fisherinformation(ef)) atol = 1e-8
        end
    end

    @testset "ExponentialFamilyDistribution mean,var" begin
        for (λ, k) in Iterators.product(1:5, 1:5)
            dist = Weibull(λ, k)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end
