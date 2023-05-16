module MultinomialTest

using Test
using ExponentialFamily
using Distributions
using Random
using StableRNGs
import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure

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
        n = 10
        left = Multinomial(n, [0.2, 0.4, 0.4])
        right = Multinomial(n, [0.1, 0.3, 0.6])
        prod_dist = prod(ClosedProd(), left, right)

        d = Multinomial(n, ones(3) ./ 3)
        sample_space = unique(rand(StableRNG(1), d, 4000), dims = 2)
        sample_space = [sample_space[:, i] for i in 1:size(sample_space, 2)]

        sample_x = [[2, 5, 3], [1, 2, 7], [0, 4, 6], [1, 4, 5]]
        for xi in sample_x
            @test prod_dist.basemeasure(xi) == (factorial(n) / prod(factorial.(xi)))^2
            @test prod_dist.sufficientstatistics(xi) == xi
            hist_sum(x) =
                prod_dist.basemeasure(x) * exp(
                    prod_dist.naturalparameters' * prod_dist.sufficientstatistics(x) -
                    prod_dist.logpartition(prod_dist.naturalparameters)
                )
            @test sum(hist_sum(x_sample) for x_sample in sample_space) ≈ 1.0 atol = 1e-10
        end

        @test_throws AssertionError prod(
            ClosedProd(),
            Multinomial(4, [0.2, 0.4, 0.4]),
            Multinomial(5, [0.1, 0.3, 0.6])
        )
        @test_throws AssertionError prod(
            ClosedProd(),
            Multinomial(4, [0.2, 0.4, 0.4]),
            Multinomial(3, [0.1, 0.3, 0.6])
        )
    end

    @testset "natural parameters related " begin
        d1 = Multinomial(5, [0.1, 0.4, 0.5])
        d2 = Multinomial(5, [0.2, 0.4, 0.4])
        η1 = KnownExponentialFamilyDistribution(Multinomial, [log(0.1 / 0.5), log(0.4 / 0.5), 0.0], 5)
        η2 = KnownExponentialFamilyDistribution(Multinomial, [log(0.2 / 0.4), 0.0, 0.0], 5)

        @test convert(KnownExponentialFamilyDistribution, d1) == η1
        @test convert(KnownExponentialFamilyDistribution, d2) == η2

        @test convert(Distribution, η1) ≈ d1
        @test convert(Distribution, η2) ≈ d2

        @test logpartition(η1) == 3.4657359027997265
        @test logpartition(η2) == 4.5814536593707755

        @test basemeasure(d1, [1, 2, 2]) == 30.0
        @test basemeasure(d2, [1, 2, 2]) == 30.0
        @test basemeasure(η1, [1, 2, 2]) == 30.0
        @test basemeasure(η2, [1, 2, 2]) == 30.0
        @test basemeasure(d1, [1, 2, 2]) == basemeasure(η1, [1, 2, 2])
        @test basemeasure(d2, [1, 2, 2]) == basemeasure(η2, [1, 2, 2])

        @test prod(η1, η2) ==
              KnownExponentialFamilyDistribution(Multinomial, [log(0.1 / 0.5) + log(0.2 / 0.4), log(0.4 / 0.5), 0.0], 5)
        @test logpdf(η1, [1, 2, 2]) == logpdf(d1, [1, 2, 2])
        @test logpdf(η2, [1, 2, 2]) == logpdf(d2, [1, 2, 2])

        @test pdf(η1, [1, 2, 2]) == pdf(d1, [1, 2, 2])
        @test pdf(η2, [1, 2, 2]) == pdf(d2, [1, 2, 2])
    end
end
end
