module NegativeBinomialTest

using Test
using ExponentialFamily
using Distributions
using Random
import StatsFuns: logit
import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure

@testset "NegativeBinomial" begin
    @testset "probvec" begin
        @test all(probvec(NegativeBinomial(2, 0.8)) .≈ (0.2, 0.8)) # check
        @test probvec(NegativeBinomial(2, 0.2)) == (0.8, 0.2)
        @test probvec(NegativeBinomial(2, 0.1)) == (0.9, 0.1)
        @test probvec(NegativeBinomial(2)) == (0.5, 0.5)
    end

    @testset "vague" begin
        @test_throws MethodError vague(NegativeBinomial)
        @test_throws MethodError vague(NegativeBinomial, 1 / 2)

        vague_dist = vague(NegativeBinomial, 5)
        @test typeof(vague_dist) <: NegativeBinomial
        @test probvec(vague_dist) == (0.5, 0.5)
    end

    @testset "prod" begin
        @test prod(ClosedProd(), NegativeBinomial(2, 0.7), NegativeBinomial(2, 0.1)) ≈ NegativeBinomial(2, 7 / 34)

        @test prod(ClosedProd(), NegativeBinomial(3, 0.9), NegativeBinomial(3, 0.4)) ≈ NegativeBinomial(3, 6 / 7)

        @test prod(ClosedProd(), NegativeBinomial(4, 0.8), NegativeBinomial(4, 0.2)) ≈ NegativeBinomial(4, 0.5)

        @test_throws AssertionError prod(
            ClosedProd(),
            NegativeBinomial(4, 0.8),
            NegativeBinomial(5, 0.2)
        )
        @test_throws AssertionError prod(
            ClosedProd(),
            NegativeBinomial(5, 0.8),
            NegativeBinomial(4, 0.2)
        )
    end

    @testset "naturalparameter related NegativeBinomial" begin
        d1 = NegativeBinomial(5, 1 / 3)
        d2 = NegativeBinomial(5, 1 / 2)
        η1 = KnownExponentialFamilyDistribution(NegativeBinomial, [log(1 / 3)], 5)
        η2 = KnownExponentialFamilyDistribution(NegativeBinomial, [log(1 / 2)], 5)

        @test convert(KnownExponentialFamilyDistribution, d1) == η1
        @test convert(KnownExponentialFamilyDistribution, d2) == η2

        @test convert(Distribution, η1) ≈ d1
        @test convert(Distribution, η2) ≈ d2

        η3 = KnownExponentialFamilyDistribution(NegativeBinomial, [log(0.1)], 5)
        η4 = KnownExponentialFamilyDistribution(NegativeBinomial, [log(0.2)], 10)

        @test logpartition(η3) ≈ -5.0 * log(1 - 0.1)
        @test logpartition(η4) ≈ -10.0 * log(1 - 0.2)

        @test basemeasure(d1, 5) == binomial(9, 5)
        @test basemeasure(d2, 2) == binomial(6, 2)
        @test basemeasure(η1, 5) == basemeasure(d1, 5)
        @test basemeasure(η2, 2) == basemeasure(d2, 2)

        @test prod(η1, η2) == KnownExponentialFamilyDistribution(NegativeBinomial, [log(1 / 3) + log(1 / 2)], 5)

        @test logpdf(η1, 2) == logpdf(d1, 2)
        @test logpdf(η2, 3) == logpdf(d2, 3)

        @test pdf(η1, 2) == pdf(d1, 2)
        @test pdf(η2, 4) == pdf(d2, 4)
    end
end
end
