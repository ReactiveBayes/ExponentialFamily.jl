module BinomialTest

using Test
using ExponentialFamily
using Distributions
using Random
import StatsFuns: logit
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, basemeasure

@testset "Binomial" begin
    @testset "probvec" begin
        @test all(probvec(Binomial(2, 0.8)) .≈ (0.2, 0.8)) # check
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

    @testset "prod" begin
        @test prod(ProdAnalytical(), Binomial(2, 0.7), Binomial(2, 0.1)) ≈ Binomial(2, 7 / 34)

        @test prod(ProdAnalytical(), Binomial(3, 0.9), Binomial(3, 0.4)) ≈ Binomial(3, 6 / 7)

        @test prod(ProdAnalytical(), Binomial(4, 0.8), Binomial(4, 0.2)) ≈ Binomial(4, 0.5)

        @test_throws AssertionError prod(
            ProdAnalytical(),
            Binomial(4, 0.8),
            Binomial(5, 0.2)
        )
        @test_throws AssertionError prod(
            ProdAnalytical(),
            Binomial(5, 0.8),
            Binomial(4, 0.2)
        )
    end

    @testset "naturalparameter related Binomial" begin
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

        @test lognormalizer(η3) ≈ 5.0
        @test lognormalizer(η4) ≈ 10.0

        @test basemeasure(d1, 5) == 1
        @test basemeasure(d2, 2) == 10
        @test basemeasure(η1, 5) == basemeasure(d1, 5)
        @test basemeasure(η2, 2) == basemeasure(d2, 2)

        @test η1 + η2 == ExponentialFamilyDistribution(Binomial, [logit(1 / 3) + logit(1 / 2)], 5)
        @test η1 - η2 == ExponentialFamilyDistribution(Binomial, [logit(1 / 3) - logit(1 / 2)], 5)
        @test η3 - η4 ==
              [ExponentialFamilyDistribution(Binomial, [log(exp(1) - 1)], 5), ExponentialFamilyDistribution(Binomial, [-log(exp(1) - 1)], 10)]
        @test η1 + η2 - η2 ≈ η1
        @test η1 + η2 - η1 ≈ η2
        @test η3 + η4 == [η3, η4]

        @test logpdf(η1, 2) == logpdf(d1, 2)
        @test logpdf(η2, 3) == logpdf(d2, 3)

        @test pdf(η1, 2) == pdf(d1, 2)
        @test pdf(η2, 4) == pdf(d2, 4)
    end
end
end
