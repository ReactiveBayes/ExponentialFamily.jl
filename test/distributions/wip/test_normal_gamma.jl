module NormalGammaTests

using Distributions
using Test
using StatsFuns: logmvgamma
using LinearAlgebra
using ForwardDiff
using ExponentialFamily
using StableRNGs
import ExponentialFamily:
    NormalGamma, ExponentialFamilyDistribution, params, location
import ExponentialFamily:
    scale, dim, getnaturalparameters, tiny, logpartition, cholinv, MvNormalMeanPrecision, sufficientstatistics,
    fisherinformation
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
            ef = convert(ExponentialFamilyDistribution, dist)
            dist_converted = convert(Distribution, ef)
            @test dist_converted ≈ dist
            @test getnaturalparameters(ef) ≈ [s * m, -s / 2, a - 1 / 2, -b - s * m^2 / 2]
        end
    end

    @testset "exponential family functions" begin
        rng = StableRNG(42)
        for i in 1:0.1:5
            m = rand(rng)
            s = rand(rng)
            a = i
            b = i^2
            dist = NormalGamma(m, s, a, b)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test pdf(dist, [m, s]) ≈ normal_gamma_pdf(m, s, m, s, a, b)
            @test logpdf(dist, [m, s]) ≈ log(normal_gamma_pdf(m, s, m, s, a, b))
        end
    end
    transformation(η) = [η[1] / (-2η[2]), -2η[2], η[3] + 1 / 2, -η[4] + (η[1]^2 / 4η[2])]
    @testset "fisher information (naturalparameters)" begin
        rng = StableRNG(42)
        for i in 1:0.1:5
            m = rand(rng)
            s = rand(rng)
            a = i
            b = i
            dist = NormalGamma(m, s, a, b)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)
            f_logpartion = (η) -> logpartition(ExponentialFamilyDistribution(NormalGamma, η))
            autograd_inforamation_matrix = (η) -> ForwardDiff.hessian(f_logpartion, η)
            fef = fisherinformation(ef)
            fdist = fisherinformation(dist)
            J = ForwardDiff.jacobian(transformation, η)
            @test fef ≈ autograd_inforamation_matrix(η)
            @test J' * fdist * J ≈ fef
        end
    end

    @testset "sampling" begin
        rng = StableRNG(42)
        for i in 1:2:10, j in 1:2:10
            # Parameters for the Normal-Gamma distribution
            μ = 2.0
            λ = 1.0
            α = 3.0 + i
            β = 2.0 + j

            # Create a Normal-Gamma distribution
            dist = NormalGamma(μ, λ, α, β)

            # Generate a large number of samples
            n_samples = 100000
            samples = rand(rng, dist, n_samples)

            # Calculate the sample means and variances
            sample_mean_x = mean(x -> x[1], samples)
            sample_var_x = var(getindex.(samples, 1))
            sample_mean_v = mean(x -> x[2], samples)
            sample_var_v = var(getindex.(samples, 2))

            # Expected means and variances
            expected_mean_x = μ
            expected_var_x = β / (α - 1) / λ
            expected_mean_v = α / β
            expected_var_v = α / β^2

            # Compare the sample means and variances to the expected values
            @test isapprox(sample_mean_x, expected_mean_x, atol = 0.1)
            @test isapprox(sample_var_x, expected_var_x, atol = 0.1)
            @test isapprox(sample_mean_v, expected_mean_v, atol = 0.1)
            @test isapprox(sample_var_v, expected_var_v, atol = 0.1)
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
            ef1 = convert(ExponentialFamilyDistribution, dist1)
            ef2 = convert(ExponentialFamilyDistribution, dist2)
            @test prod(ClosedProd(), dist1, dist2) == convert(Distribution, prod(ClosedProd(), ef1, ef2))
        end
    end

    @testset "ExponentialFamilyDistribution mean,var" begin
        for i in 1:10, j in 1:10
            # Parameters for the Normal-Gamma distribution
            μ = 2.0
            λ = 1.0
            α = 3.0 + i
            β = 2.0 + j

            # Create a Normal-Gamma distribution
            dist = NormalGamma(μ, λ, α, β)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test cov(dist) ≈ cov(ef) atol = 1e-8
        end
    end
end

end
