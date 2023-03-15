module BinomialTest

using Test
using ExponentialFamily
using Distributions
using Random
import ExponentialFamily: NaturalParameters, get_params, basemeasure

@testset "Binomial" begin
    @testset "probvec" begin
        @test probvec(Binomial(2, 0.8)) == (0.2, 0.8)
        @test probvec(Binomial(2, 0.2)) == (0.8, 0.2)
        @test probvec(Binomial(2, 0.1)) == (0.9, 0.1)
        @test probvec(Binomial(2)) == (0.5, 0.5)
    end

    @testset "vague" begin
        @test_throws MethodError vague(Multinomial)
        @test_throws MethodError vague(Multinomial, 4)

        vague_dist = vague(Binomial, 5)
        @test typeof(vague_dist) <: Binomial
        @test probvec(vague_dist) == (0.5, 0.5)

    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Binomial(2, 0.7), Binomial(2, 0.1)) ≈ Binomial(2, 7/34)

        @test prod(ProdAnalytical(), Binomial(3, 0.9), Binomial(3, 0.3)) ≈ Binomial(2, 0.3)
        
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
        d1 = Binomial(5, 1/3)
        d2 = Binomial(5, 1/2)
        η1 = NaturalParameters(Binomial, logit(1/3), 5)
        η2 = NaturalParameters(Binomial, logit(1/2), 5)

        @test convert(NaturalParameters, d1) == η1
        @test convert(NaturalParameters, d2) == η2

        @test convert(Distribution, η1) ≈ d1
        @test convert(Distribution, η2) ≈ d2

        @test lognormalizer(η1) == 0.0
        @test lognormalizer(η2) == 0.0

        @test basemeasure(d1, [1, 2, 2]) == 30.0
        @test basemeasure(d2, [1, 2, 2]) == 30.0
        @test basemeasure(η1, [1, 2, 2]) == 30.0
        @test basemeasure(η2, [1, 2, 2]) == 30.0
        @test basemeasure(d1, [1, 2, 2]) == basemeasure(η1, [1, 2, 2])
        @test basemeasure(d2, [1, 2, 2]) == basemeasure(η2, [1, 2, 2])

        @test η1 + η2 == NaturalParameters(Multinomial, [log(0.1) + log(0.2), 2log(0.4), log(0.5) + log(0.4)], 5)
        @test η1 - η2 == NaturalParameters(Multinomial, [log(0.1) - log(0.2), 0.0, log(0.5) - log(0.4)], 5)
        @test η1 + η2 - η2 ≈ η1
        @test η1 + η2 - η1 ≈ η2
        η3 = NaturalParameters(Multinomial, [log(0.1), log(0.4), log(0.5)], 5)
        η4 = NaturalParameters(Multinomial, [log(0.1), log(0.4), log(0.5)], 6)
        @test η3 + η4 == [η3, η4]
        @test logpdf(η1, [1, 2, 2]) == logpdf(d1, [1, 2, 2])
        @test logpdf(η2, [1, 2, 2]) == logpdf(d2, [1, 2, 2])

        @test pdf(η1, [1, 2, 2]) == pdf(d1, [1, 2, 2])
        @test pdf(η2, [1, 2, 2]) == pdf(d2, [1, 2, 2])
    end
end
end
