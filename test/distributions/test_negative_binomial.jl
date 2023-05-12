module NegativeBinomialTest

using Test
using ExponentialFamily
using Distributions
using Random
import StatsFuns: logit
import DomainSets: NaturalNumbers
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
        left = NegativeBinomial(10, 0.5)
        right = NegativeBinomial(15, 0.3)
        prod_dist = prod(ClosedProd(), left, right)

        η_left = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, left)))
        η_right = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, right)))

        sample_points = collect(1:5)
        for x in sample_points
            rleft, rright = Integer(first(params(left))), Integer(first(params(right)))

            @test prod_dist.basemeasure(x) == (binomial(BigInt(x + rleft - 1), x) * binomial(BigInt(x + rright - 1), x))
            @test prod_dist.sufficientstatistics(x) == x

            hist_sum(x) =
                prod_dist.basemeasure(x) * exp(
                    prod_dist.sufficientstatistics(x) * prod_dist.naturalparameters[1] -
                    prod_dist.logpartition(prod_dist.naturalparameters[1])
                )
            @test sum(hist_sum(x) for x in 0:15) ≈ 1.0
        end
    end

    @testset "natural parameters related" begin
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

        @test isproper(KnownExponentialFamilyDistribution(NegativeBinomial, [0], 5)) == true
        for x in 1:10
            ef_proper = KnownExponentialFamilyDistribution(NegativeBinomial, [-x], 5)
            ef_improper = KnownExponentialFamilyDistribution(NegativeBinomial, [x], 5)
            @test isproper(ef_proper) == true
            @test isproper(ef_improper) == false
        end
    end
end
end
