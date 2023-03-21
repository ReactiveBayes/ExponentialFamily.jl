module ContinuousBernoulliTest

using Test
using ExponentialFamily
using Distributions
using Random
using StatsFuns
import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, compute_logscale, logpartition, basemeasure

@testset "ContinuousBernoulli" begin
    @testset "vague" begin
        d = vague(ContinuousBernoulli)

        @test typeof(d) <: ContinuousBernoulli
        @test mean(d) === 0.5
        @test succprob(d) === 0.5
        @test failprob(d) === 0.5
    end

    @testset "prod ContinuousBernoulli-ContinuousBernoulli" begin
        @test prod(ClosedProd(), ContinuousBernoulli(0.5), ContinuousBernoulli(0.5)) ≈ ContinuousBernoulli(0.5)
        @test prod(ClosedProd(), ContinuousBernoulli(0.1), ContinuousBernoulli(0.6)) ≈
              ContinuousBernoulli(0.14285714285714285)
        @test prod(ClosedProd(), ContinuousBernoulli(0.78), ContinuousBernoulli(0.05)) ≈
              ContinuousBernoulli(0.1572580645161291)
    end

    @testset "probvec" begin
        @test probvec(ContinuousBernoulli(0.5)) === (0.5, 0.5)
        @test probvec(ContinuousBernoulli(0.3)) === (0.7, 0.3)
        @test probvec(ContinuousBernoulli(0.6)) === (0.4, 0.6)
    end

    @testset "KnownExponentialFamilyDistribution" begin
        @test logpartition(convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.5))) ≈ log(2)
        @test logpartition(convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.2))) ≈ log((-3 / 4) / log(1 / 4))
        b_99 = ContinuousBernoulli(0.99)
        for i in 1:9
            b = ContinuousBernoulli(i / 10.0)
            bnp = convert(KnownExponentialFamilyDistribution, b)
            @test convert(Distribution, bnp) ≈ b
            @test logpdf(bnp, 1) ≈ logpdf(b, 1)
            @test logpdf(bnp, 0) ≈ logpdf(b, 0)

            @test convert(KnownExponentialFamilyDistribution, b) == KnownExponentialFamilyDistribution(ContinuousBernoulli, [logit(i / 10.0)])

            # @test prod(ClosedProd(), convert(Distribution, convert(KnownExponentialFamilyDistribution, b_99) - bnp), b) ≈ b_99
        end
        @test isproper(KnownExponentialFamilyDistribution(ContinuousBernoulli, [10])) === true
        @test basemeasure(b_99, 0.1) == 1.0
        @test basemeasure(KnownExponentialFamilyDistribution(ContinuousBernoulli, [10]), 0.2) == 1.0

        @testset "prod(::KnownExponentialFamilyDistribution{ContinuousBernoulli}, ::KnownExponentialFamilyDistribution{ContinuousBernoulli})" begin
            left = convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.5))
            right = convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.6))
            @test prod(left ,right) == convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.6))
        end
    end

    @testset "rand" begin
        dist = ContinuousBernoulli(0.3)
        nsamples = 1000
        rng = collect(1:100)
        for i in 1:10
            samples = rand(MersenneTwister(rng[i]), dist, nsamples)
            mestimated = mean(samples)
            weights = ones(nsamples) / nsamples
            @test isapprox(mestimated, mean(dist), atol = 1e-1)
            @test isapprox(
                sum(weight * (sample - mestimated)^2 for (sample, weight) in (samples, weights)),
                var(dist),
                atol = 1e-1
            )
        end
    end
end

end
