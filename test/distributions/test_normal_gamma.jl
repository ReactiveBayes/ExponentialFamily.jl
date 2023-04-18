module NormalGammaTests

using Distributions
using Test
using StatsFuns: logmvgamma
using LinearAlgebra
using ForwardDiff
using ExponentialFamily
import ExponentialFamily:
    NormalGamma, KnownExponentialFamilyDistribution, params, location
import ExponentialFamily:
    scale, dim, getnaturalparameters, tiny, logpartition, cholinv, MvNormalMeanPrecision, sufficientstatistics
using Distributions
using Random

function normal_gamma_pdf(x, τ, μ, λ, α, β)
    return pdf(NormalMeanPrecision(μ, λ * τ), x) * pdf(GammaShapeRate(α, β), τ)
end

@testset "NormalGamma" begin
    @testset "common" begin
        m = rand()
        s, a, b = 1.0, 0.1, 3.0
        dist = NormalGamma(m, s, a, b)
        @test params(dist) == (m, s, a, b)
        @test location(dist) == m
        @test scale(dist) == s
        @test shape(dist) == a
        @test rate(dist) == b
    end

    @testset "conversions" begin
        for i in 1:0.1:5
            m = rand()
            s = rand()
            a = i
            b = i
            dist = NormalGamma(m, s, a, b)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            @test getnaturalparameters(ef) ≈ [s * m, -s / 2, a - 1 / 2, -b - s * m^2 / 2]
        end
    end

    @testset "exponential family functions" begin
        for i in 1:0.1:5
            m = rand()
            s = rand()
            a = i
            b = i
            dist = NormalGamma(m, s, a, b)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            @test pdf(dist, [m, s]) ≈ normal_gamma_pdf(m, s, m, s, a, b)
            @test logpdf(dist, [m, s]) ≈ log(normal_gamma_pdf(m, s, m, s, a, b))
        end
    end

    @testset "prod" begin
        for _ in 1:10
            m1 = rand()
            m2 = rand()
            a1 = rand()
            a2 = rand()
            dist1 = NormalGamma(m1, 1.0, a1, 1.0)
            dist2 = NormalGamma(m2, 1.0, a2, 1.0)
            ef1 = convert(KnownExponentialFamilyDistribution, dist1)
            ef2 = convert(KnownExponentialFamilyDistribution, dist2)
            @test prod(ClosedProd(), dist1, dist2) == convert(Distribution, prod(ClosedProd(), ef1, ef2))
        end
    end
end

end
