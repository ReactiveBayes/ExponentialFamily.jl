module ContinuousBernoulliTest

using Test
using ExponentialFamily
using Distributions
using Random
using StatsFuns
using ForwardDiff
import ExponentialFamily:
    KnownExponentialFamilyDistribution, getnaturalparameters, compute_logscale, logpartition, basemeasure,
    fisherinformation, isvague

@testset "ContinuousBernoulli" begin
    @testset "vague" begin
        d = vague(ContinuousBernoulli)

        @test typeof(d) <: ContinuousBernoulli
        @test mean(d) === 0.5
        @test succprob(d) === 0.5
        @test failprob(d) === 0.5
    end

    @testset "probvec" begin
        @test probvec(ContinuousBernoulli(0.5)) === (0.5, 0.5)
        @test probvec(ContinuousBernoulli(0.3)) === (0.7, 0.3)
        @test probvec(ContinuousBernoulli(0.6)) === (0.4, 0.6)
    end

    @testset "natural parameters related" begin
        @test logpartition(convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.5))) ≈ log(2)
        @test logpartition(convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.2))) ≈
              log((-3 / 4) / log(1 / 4))
        b_99 = ContinuousBernoulli(0.99)
        for i in 1:9
            b = ContinuousBernoulli(i / 10.0)
            bnp = convert(KnownExponentialFamilyDistribution, b)
            @test convert(Distribution, bnp) ≈ b
            @test logpdf(bnp, 1) ≈ logpdf(b, 1)
            @test logpdf(bnp, 0) ≈ logpdf(b, 0)

            @test convert(KnownExponentialFamilyDistribution, b) ==
                  KnownExponentialFamilyDistribution(ContinuousBernoulli, [logit(i / 10.0)])
        end
        @test isproper(KnownExponentialFamilyDistribution(ContinuousBernoulli, [10])) === true
        @test basemeasure(KnownExponentialFamilyDistribution(ContinuousBernoulli, [10]), 0.2) == 1.0
    end

    @testset "prod" begin
        @test prod(ClosedProd(), ContinuousBernoulli(0.5), ContinuousBernoulli(0.5)) ≈ ContinuousBernoulli(0.5)
        @test prod(ClosedProd(), ContinuousBernoulli(0.1), ContinuousBernoulli(0.6)) ≈
              ContinuousBernoulli(0.14285714285714285)
        @test prod(ClosedProd(), ContinuousBernoulli(0.78), ContinuousBernoulli(0.05)) ≈
              ContinuousBernoulli(0.1572580645161291)

        left = convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.5))
        right = convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.6))
        @test prod(left, right) == convert(KnownExponentialFamilyDistribution, ContinuousBernoulli(0.6))
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

    @testset "fisher information" begin
        function transformation(params)
            return logistic(params[1])
        end

        for κ in 0.000001:0.01:0.49
            dist = ContinuousBernoulli(κ)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(ContinuousBernoulli, η))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test first(fisherinformation(ef)) ≈ first(autograd_information(η)) atol = 1e-9
            J = ForwardDiff.gradient(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) atol = 1e-9
        end

        for κ in 0.51:0.01:0.99
            dist = ContinuousBernoulli(κ)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(ContinuousBernoulli, η))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test first(fisherinformation(ef)) ≈ first(autograd_information(η)) atol = 1e-9
            J = ForwardDiff.gradient(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) atol = 1e-9
        end

        for κ in 0.499:0.0001:0.50001
            dist = ContinuousBernoulli(κ)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            J = ForwardDiff.gradient(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) atol = 1e-9
        end
    end

    @testset "KnownExponentialFamilyDistribution mean var" begin
        for ν in 0.1:0.1:0.99
            dist = ContinuousBernoulli(ν)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end

end
