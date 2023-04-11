module WeibullTest

using Test
using ExponentialFamily
using Distributions
using Random
using HCubature
using SpecialFunctions

import ExponentialFamily: KnownExponentialFamilyDistribution, ExponentialFamilyDistribution, getnaturalparameters,
    getsufficientstatistics, getlogpartition, getbasemeasure
import ExponentialFamily: basemeasure, isproper

@testset "Weibull" begin

    # Weibull comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "convert" begin
        for λ in 0.5:0.5:10, k in 0.5:0.5:10
            @test convert(KnownExponentialFamilyDistribution, Weibull(k, λ)) ≈
                  KnownExponentialFamilyDistribution(Weibull, [-(1 / λ)^(k)], k)
        end
    end

    @testset "logpartition" begin
        @test logpartition(KnownExponentialFamilyDistribution(Weibull, [-1], 1)) ≈ 0
    end

    @testset "isproper" begin
        for η in -10:0.5:10, k in 0.5:0.5:10
            @test isproper(KnownExponentialFamilyDistribution(Weibull, [η], k)) == (η < 0)
        end
    end

    @testset "basemeasure" begin
        for η in -10:0.5:-0.5, k in 0.5:0.5:10, x in 0.5:0.5:10
            @test basemeasure(KnownExponentialFamilyDistribution(Weibull, [η], k), x) ≈ x^(k - 1)
        end
    end

    @testset "same k prod" begin
        for η in -10:0.5:-0.5, k in 1.0:0.5:10, x in 0.5:0.5:10
            ef_left = convert(Distribution, KnownExponentialFamilyDistribution(Weibull, [η], k))
            ef_right = convert(Distribution, KnownExponentialFamilyDistribution(Weibull, [-η^2], k))
            res = prod(ClosedProd(), ef_left, ef_right)
            @test getbasemeasure(res)(x) == x^(2 * (k - 1))
            @test getsufficientstatistics(res)(x) == x^k
            @test getlogpartition(res)(η - η^2) ==
                  log(abs(η - η^2)^(1 / k)) + loggamma(2 - 1 / k) - 2 * log(abs(η - η^2)) - log(k)
            @test getnaturalparameters(res) ≈ [η - η^2]
            @test first(hquadrature(x -> pdf(res, tan(x * pi / 2)) * (pi / 2) * (1 / cos(pi * x / 2))^2, 0.0, 1.0)) ≈
                  1.0
        end
    end

    @testset "different k prod" begin
        for η in -12:4:-0.5, k in 1.0:4:10, x in 0.5:4:10
            ef_left = convert(Distribution, KnownExponentialFamilyDistribution(Weibull, [η], k * 2))
            ef_right = convert(Distribution, KnownExponentialFamilyDistribution(Weibull, [-η^2], k))
            res = prod(ClosedProd(), ef_left, ef_right)
            @test getbasemeasure(res)(x) == x^(k + k * 2 - 2)
            @test getsufficientstatistics(res)(x) == [x^(2 * k), x^k]
            @test getnaturalparameters(res) ≈ [η, -η^2]
            @test first(hquadrature(x -> pdf(res, tan(x * pi / 2)) * (pi / 2) * (1 / cos(pi * x / 2))^2, 0.0, 1.0)) ≈
                  1.0
        end
    end

    @testset "isproper" begin
        for η in -12:4:-0.5, k in 1.0:4:10
            ef_proper = KnownExponentialFamilyDistribution(Weibull, [η], k )
            ef_improper = KnownExponentialFamilyDistribution(Weibull, [-η], k )
            @test isproper(ef_proper) == true
            @test isproper(ef_improper) == false
        end
    end
end
end
