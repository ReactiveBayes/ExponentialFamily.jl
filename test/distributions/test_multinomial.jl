module MultinomialTest

using Test
using ExponentialFamily
using Distributions
using Random
import ExponentialFamily: NaturalParameters, get_params, basemeasure

@testset "Multinomial" begin
    @testset "probvec" begin
        @test probvec(Multinomial(5, [1 / 3, 1 / 3, 1 / 3])) == [1 / 3, 1 / 3, 1 / 3]
        @test probvec(Multinomial(3, [0.2, 0.2, 0.4, 0.1, 0.1])) == [0.2, 0.2, 0.4, 0.1, 0.1]
        @test probvec(Multinomial(2, [0.5, 0.5])) == [0.5, 0.5]
    end

    @testset "vague" begin
        @test_throws MethodError vague(Multinomial)
        @test_throws MethodError vague(Multinomial, 4)

        vague_dist1 = vague(Multinomial, 5, 4)
        @test typeof(vague_dist1) <: Multinomial
        @test probvec(vague_dist1) == [1 / 4, 1 / 4, 1 / 4, 1 / 4]

        vague_dist2 = vague(Multinomial, 3, 5)
        @test typeof(vague_dist2) <: Multinomial
        @test probvec(vague_dist2) == [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Multinomial(4, [0.2, 0.4, 0.4]), Multinomial(4, [0.1, 0.3, 0.6])) ≈
              Multinomial(4, [0.05263157894736842, 0.3157894736842105, 0.631578947368421])

        @test prod(ProdAnalytical(), Multinomial(3, [0.6, 0.4]), Multinomial(3, [0.3, 0.7])) ≈
              Multinomial(3, [0.3913043478260869, 0.6086956521739131])

        @test prod(
            ProdAnalytical(),
            Multinomial(10, [1 / 4, 1 / 4, 1 / 4, 1 / 4]),
            Multinomial(10, [0.1, 0.4, 0.3, 0.2])
        ) ==
              Multinomial(10, [0.1, 0.4, 0.3, 0.2])

        @test_throws AssertionError prod(
            ProdAnalytical(),
            Multinomial(4, [0.2, 0.4, 0.4]),
            Multinomial(5, [0.1, 0.3, 0.6])
        )
        @test_throws AssertionError prod(
            ProdAnalytical(),
            Multinomial(4, [0.2, 0.4, 0.4]),
            Multinomial(3, [0.1, 0.3, 0.6])
        )
    end

    @testset "naturalparameter related Multinomial" begin
        d1 = Multinomial(5, [0.1, 0.4, 0.5])
        d2 = Multinomial(5, [0.2, 0.4, 0.4])
        η1 = NaturalParameters(Multinomial, [log(0.1), log(0.4), log(0.5)], 5)
        η2 = NaturalParameters(Multinomial, [log(0.2), log(0.4), log(0.4)], 5)

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
