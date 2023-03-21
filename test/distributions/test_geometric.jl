module GeometricTest

using Test
using Random
using Distributions
using ExponentialFamily
import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure

@testset "Geometric" begin
    @testset "Geometric vague" begin
        d = Geometric(0.6)

        @test Geometric() == Geometric(0.5)
        @test typeof(vague(Geometric)) <: Geometric
        @test vague(Geometric) == Geometric(1e-12)
        @test succprob(d) == 0.6
        @test failprob(d) == 0.4
        @test probvec(d) == (0.4, 0.6)
    end

    @testset "Geometric prod" begin
        @test prod(ClosedProd(), Geometric(0.5), Geometric(0.6)) == Geometric(0.8)
        @test prod(ClosedProd(), Geometric(0.3), Geometric(0.8)) == Geometric(0.8600000000000001)
        @test prod(ClosedProd(), Geometric(0.5), Geometric(0.5)) == Geometric(0.75)
    end

    @testset "naturalparameter related Geometric" begin
        d1 = Geometric(0.6)
        d2 = Geometric(0.3)
        η1 = KnownExponentialFamilyDistribution(Geometric, [log(1 - 0.6)])
        η2 = KnownExponentialFamilyDistribution(Geometric, [log(1 - 0.3)])

        @test convert(Geometric, η1) ≈ d1
        @test convert(Geometric, η2) ≈ d2

        @test convert(KnownExponentialFamilyDistribution, d1) == η1
        @test convert(KnownExponentialFamilyDistribution, d2) == η2

        @test logpartition(η1) ≈ -log(0.6)
        @test logpartition(η2) ≈ -log(0.3)

        @test basemeasure(d1, 5) == 1.0
        @test basemeasure(d2, 1) == 1.0
        @test basemeasure(η1, 4) == 1.0
        @test basemeasure(η2, 2) == 1.0

        @test prod(η1 , η2) == KnownExponentialFamilyDistribution(Geometric, [log(0.4) + log(0.7)])
        # @test η1 - η2 == KnownExponentialFamilyDistribution(Geometric, [log(0.4) - log(0.7)])

        @test logpdf(η1, 3) == logpdf(d1, 3)
        @test logpdf(η2, 3) == logpdf(d2, 3)

        @test pdf(η1, 3) == pdf(d1, 3)
        @test pdf(η2, 3) == pdf(d2, 3)

        @test isproper(KnownExponentialFamilyDistribution(Geometric, [log(0.6)])) == true
        @test isproper(KnownExponentialFamilyDistribution(Geometric, [1.3])) == false
    end
end
end
