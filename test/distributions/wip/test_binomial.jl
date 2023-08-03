module BinomialTest

using Test
using ExponentialFamily, LinearAlgebra
using Distributions
using Random
using ForwardDiff
import StatsFuns: logit, logistic
import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation, sufficientstatistics

@testset "Binomial" begin
    @testset "probvec" begin
        @test all(probvec(Binomial(2, 0.8)) .≈ (0.2, 0.8))
        @test probvec(Binomial(2, 0.2)) == (0.8, 0.2)
        @test probvec(Binomial(2, 0.1)) == (0.9, 0.1)
        @test probvec(Binomial(2)) == (0.5, 0.5)
    end

    @testset "vague" begin
        @test_throws MethodError vague(Binomial)
        @test_throws MethodError vague(Binomial, 1 / 2)

        vague_dist = vague(Binomial, 5)
        @test typeof(vague_dist) <: Binomial
        @test probvec(vague_dist) == (0.5, 0.5)
    end

    @testset "natural parameters related" begin
        d1 = Binomial(5, 1 / 3)
        d2 = Binomial(5, 1 / 2)
        η1 = ExponentialFamilyDistribution(Binomial, [logit(1 / 3)], 5)
        η2 = ExponentialFamilyDistribution(Binomial, [logit(1 / 2)], 5)

        @test convert(ExponentialFamilyDistribution, d1) == η1
        @test convert(ExponentialFamilyDistribution, d2) == η2

        @test convert(Distribution, η1) ≈ d1
        @test convert(Distribution, η2) ≈ d2

        η3 = ExponentialFamilyDistribution(Binomial, [log(exp(1) - 1)], 5)
        η4 = ExponentialFamilyDistribution(Binomial, [log(exp(1) - 1)], 10)

        @test logpartition(η3) ≈ 5.0
        @test logpartition(η4) ≈ 10.0

        @test logpdf(η1, 2) ≈ logpdf(d1, 2)
        @test logpdf(η2, 3) ≈ logpdf(d2, 3)

        @test pdf(η1, 2) ≈ pdf(d1, 2)
        @test pdf(η2, 4) ≈ pdf(d2, 4)

        binomialef = ExponentialFamilyDistribution(Binomial, [logit(0.3)], 10)
        @test sufficientstatistics(binomialef, 1) == [1]
        @test sufficientstatistics(binomialef, 7) == [7]
        # @test_throws AssertionError sufficientstatistics(binomialef, 11)
        # @test_throws AssertionError sufficientstatistics(binomialef, 1.1)
    end

    @testset "prod ExponentialFamilyDistribution" begin
        for nleft in 1:2, pleft in 0.01:0.3:0.99
            left = Binomial(nleft, pleft)
            efleft = convert(ExponentialFamilyDistribution, left)
            for nright in 1:1, pright in 0.01:0.3:0.99
                right = Binomial(nright, pright)
                efright = convert(ExponentialFamilyDistribution, right)
                prod_dist = prod(efleft, efright)
                hist_sum(x) =
                    prod_dist.basemeasure(x) * exp(
                        dot(prod_dist.sufficientstatistics(x), prod_dist.naturalparameters) -
                        prod_dist.logpartition(prod_dist.naturalparameters)
                    )
                @test sum(hist_sum(x) for x in 0:max(nleft, nright)) ≈ 1.0 atol = 1e-9
                sample_points = collect(1:max(nleft, nright))
                for x in sample_points
                    @test prod_dist.basemeasure(x) == (binomial(nleft, x) * binomial(nright, x))
                    @test prod_dist.sufficientstatistics(x) == [x]
                end
            end
        end
    end

    @testset "prod Distribution" begin
        for nleft in 1:15, pleft in 0.01:0.3:0.99
            left = Binomial(nleft, pleft)
            for nright in 1:10, pright in 0.01:0.3:0.99
                right = Binomial(nright, pright)
                prod_dist = prod(ClosedProd(), left, right)
                hist_sum(x) =
                    prod_dist.basemeasure(x) * exp(
                        dot(prod_dist.sufficientstatistics(x), prod_dist.naturalparameters) -
                        prod_dist.logpartition(prod_dist.naturalparameters[1])
                    )
                @test sum(hist_sum(x) for x in 0:max(nleft, nright)) ≈ 1.0 atol = 1e-9
                sample_points = collect(1:max(nleft, nright))
                for x in sample_points
                    @test prod_dist.basemeasure(x) == (binomial(nleft, x) * binomial(nright, x))
                    @test prod_dist.sufficientstatistics(x) == [x]
                end
            end
        end
    end

    @testset "fisher information" begin
        function transformation(params)
            return logistic(params[1])
        end

        for n in 2:10, κ in 0.01:0.1:1.0
            dist = Binomial(n, κ)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Binomial, η, n))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-8
            J = ForwardDiff.gradient(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ first(fisherinformation(ef)) atol = 1e-8
        end
    end

    @testset "ExponentialFamilyDistribution mean var" begin
        for n in 2:10, κ in 0.01:0.1:1.0
            dist = Binomial(n, κ)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end