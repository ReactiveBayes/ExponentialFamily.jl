module InverseWishartTest

using Test
using ExponentialFamily
using Distributions
using Random
using LinearAlgebra
using StableRNGs
using ForwardDiff

import ExponentialFamily: InverseWishartImproper, KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure,
    fisherinformation, logpartition, reconstructargument!, as_vec
import Distributions: pdf!
import StatsFuns: logmvgamma

function logpartition(ef::KnownExponentialFamilyDistribution{T}, ηvec::Vector{F}) where {T, F <: Real}
    ηef = getnaturalparameters(ef)
    reconstructargument!(ηef, ηef, ηvec)
    return logpartition(KnownExponentialFamilyDistribution(T, ηef))
end

function transformation(params)
    η1, η2 = params[1], params[2:end]
    p = Int(sqrt(length(η2)))
    η2 = reshape(η2, (p, p))
    return [-(2 * η1 + p + 1); as_vec(-2 * η2)]
end

@testset "InverseWishartImproper" begin
    @testset "common" begin
        @test InverseWishartImproper <: Distribution
        @test InverseWishartImproper <: ContinuousDistribution
        @test InverseWishartImproper <: MatrixDistribution

        @test value_support(InverseWishartImproper) === Continuous
        @test variate_form(InverseWishartImproper) === Matrixvariate
    end

    @testset "statistics" begin
        rng = StableRNG(42)
        # ν > dim(d) + 1
        for ν in 4:10
            L = randn(rng, ν - 2, ν - 2)
            S = L * L'
            d = InverseWishartImproper(ν, S)

            @test mean(d) == mean(InverseWishart(params(d)...))
            @test mode(d) == mode(InverseWishart(params(d)...))
        end

        # ν > dim(d) + 3
        for ν in 5:10
            L = randn(rng, ν - 4, ν - 4)
            S = L * L'
            d = InverseWishartImproper(ν, S)

            @test cov(d) == cov(InverseWishart(params(d)...))
            @test var(d) == var(InverseWishart(params(d)...))
        end
    end

    @testset "vague" begin
        @test_throws MethodError vague(InverseWishartImproper)

        dims = 3
        d1 = vague(InverseWishart, dims)

        @test typeof(d1) <: InverseWishart
        ν1, S1 = params(d1)
        @test ν1 == dims + 2
        @test S1 == tiny .* diageye(dims)

        @test mean(d1) == S1

        dims = 4
        d2 = vague(InverseWishart, dims)

        @test typeof(d2) <: InverseWishart
        ν2, S2 = params(d2)
        @test ν2 == dims + 2
        @test S2 == tiny .* diageye(dims)

        @test mean(d2) == S2
    end

    @testset "entropy" begin
        @test entropy(
            InverseWishartImproper(
                2.0,
                [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
            )
        ) ≈ 10.111427477184794
        @test entropy(InverseWishartImproper(5.0, diageye(4))) ≈ 8.939145914882221
    end

    @testset "convert" begin
        rng = StableRNG(42)
        for ν in 2:10
            L = randn(rng, ν, ν)
            S = L * L'
            d = InverseWishartImproper(ν, S)
            @test convert(InverseWishart, d) == InverseWishart(ν, S)
        end
    end

    @testset "mean(::typeof(logdet))" begin
        rng = StableRNG(123)
        ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(logdet, InverseWishartImproper(ν, S)), mean(logdet.(samples)), atol = 1e-2)

        ν, S = 4.0, diageye(3)
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(logdet, InverseWishartImproper(ν, S)), mean(logdet.(samples)), atol = 1e-2)
    end

    @testset "mean(::typeof(inv))" begin
        rng = StableRNG(321)
        ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(inv, InverseWishartImproper(ν, S)), mean(inv.(samples)), atol = 1e-2)
        @test isapprox(mean(cholinv, InverseWishartImproper(ν, S)), mean(cholinv.(samples)), atol = 1e-2)

        ν, S = 4.0, diageye(3)
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(inv, InverseWishartImproper(ν, S)), mean(inv.(samples)), atol = 1e-2)
        @test isapprox(mean(cholinv, InverseWishartImproper(ν, S)), mean(cholinv.(samples)), atol = 1e-2)
    end

    @testset "prod" begin
        d1 = InverseWishartImproper(3.0, diageye(2))
        d2 = InverseWishartImproper(-3.0, [0.6423504672769315 0.9203141654948761; 0.9203141654948761 1.528137747462735])

        @test prod(ClosedProd(), d1, d2) ≈
              InverseWishartImproper(3.0, [1.6423504672769313 0.9203141654948761; 0.9203141654948761 2.528137747462735])

        d1 = InverseWishartImproper(4.0, diageye(3))
        d2 = InverseWishartImproper(-2.0, diageye(3))

        @test prod(ClosedProd(), d1, d2) ≈ InverseWishartImproper(6.0, 2 * diageye(3))
    end

    @testset "rand" begin
        for d in (2, 3, 4, 5)
            v = rand() + d
            L = rand(d, d)
            S = L' * L + d * diageye(d)
            cS = copy(S)
            container1 = [zeros(d, d) for _ in 1:100]
            container2 = [zeros(d, d) for _ in 1:100]

            # Check in-place version
            @test rand!(StableRNG(321), InverseWishart(v, S), container1) ≈
                  rand!(StableRNG(321), InverseWishartImproper(v, S), container2)

            # Check that the matrix has not been corrupted
            @test all(S .=== cS)

            # Check non-inplace version
            @test rand(StableRNG(321), InverseWishart(v, S), length(container1)) ≈
                  rand(StableRNG(321), InverseWishartImproper(v, S), length(container2))
        end
    end

    @testset "pdf!" begin
        for d in (2, 3, 4, 5), n in (10, 20)
            v = rand() + d
            L = rand(d, d)
            S = L' * L + d * diageye(d)

            samples = map(1:n) do _
                L_sample = rand(d, d)
                return L_sample' * L_sample + d * diageye(d)
            end

            result = zeros(n)

            @test all(pdf(InverseWishart(v, S), samples) .≈ pdf!(result, InverseWishartImproper(v, S), samples))
        end
    end

    @testset "InverseWishartKnownExponentialFamilyDistribution" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(
                    Distribution,
                    KnownExponentialFamilyDistribution(InverseWishartImproper, [-3.0, [-i 0.0; 0.0 -i]])
                ) ≈ InverseWishart(3.0, -2([-i 0.0; 0.0 -i]))
            end
        end

        @testset "logpdf" begin
            for i in 1:10
                wishart_np = KnownExponentialFamilyDistribution(InverseWishartImproper, [-3.0, [-i 0.0; 0.0 -i]])
                distribution = InverseWishart(3.0, -2([-i 0.0; 0.0 -i]))
                @test logpdf(distribution, [1.0 0.0; 0.0 1.0]) ≈ logpdf(wishart_np, [1.0 0.0; 0.0 1.0])
                @test logpdf(distribution, [1.0 0.2; 0.2 1.0]) ≈ logpdf(wishart_np, [1.0 0.2; 0.2 1.0])
                @test logpdf(distribution, [1.0 -0.1; -0.1 3.0]) ≈ logpdf(wishart_np, [1.0 -0.1; -0.1 3.0])
            end
        end

        @testset "logpartition" begin
            @test logpartition(
                KnownExponentialFamilyDistribution(InverseWishartImproper, [-3.0, [-1.0 0.0; 0.0 -1.0]])
            ) ≈
                  logmvgamma(2, 1.5)
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(InverseWishartImproper, [3.0, [-i 0.0; 0.0 -i]])) ===
                      false
                @test isproper(KnownExponentialFamilyDistribution(InverseWishartImproper, [3.0, [i 0.0; 0.0 -i]])) ===
                      false
                @test isproper(KnownExponentialFamilyDistribution(InverseWishartImproper, [-1.0, [-i 0.0; 0.0 -i]])) ===
                      true
            end
        end

        @testset "basemeasure" begin
            for i in 1:10
                @test basemeasure(
                    KnownExponentialFamilyDistribution(InverseWishartImproper, [3.0, [-i 0.0; 0.0 -i]]),
                    rand(3, 3)
                ) == 1
            end
        end

        @testset "base operations" begin
            for i in 1:10
                np1 = KnownExponentialFamilyDistribution(InverseWishartImproper, [3.0, [i 0.0; 0.0 i]])
                np2 = KnownExponentialFamilyDistribution(InverseWishartImproper, [3.0, [2i 0.0; 0.0 2i]])
                @test prod(np1, np2) == KnownExponentialFamilyDistribution(
                    InverseWishartImproper,
                    [3.0, [i 0.0; 0.0 i]] + [3.0, [2i 0.0; 0.0 2i]]
                )
            end
        end

        @testset "fisherinformation" begin
            rng = StableRNG(42)
            for d in 2:30
                for df in d:30
                    L = randn(rng, d, d)
                    A = L * L' + 1e-8 * diageye(d)
                    dist = InverseWishart(df, A)
                    ef = convert(KnownExponentialFamilyDistribution, dist)
                    η = getnaturalparameters(ef)
                    η_vec = vcat(η[1], as_vec(η[2]))
                    fef = fisherinformation(ef)
                    fdist = fisherinformation(dist)

                    J = ForwardDiff.jacobian(transformation, η_vec)
                    @test fef ./ (J' * fdist * J) ≈ ones(size(fef))
                end
            end
        end
    end
end

end
