module GeometricTest

using Test
using Random
using Distributions
using ExponentialFamily
using StableRNGs
using ForwardDiff
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

@testset "Geometric" begin
    @testset "vague" begin
        d = Geometric(0.6)

        @test Geometric() == Geometric(0.5)
        @test typeof(vague(Geometric)) <: Geometric
        @test vague(Geometric) == Geometric(1e-12)
        @test succprob(d) == 0.6
        @test failprob(d) == 0.4
        @test probvec(d) == (0.4, 0.6)
    end

    @testset "prod" begin
        @test prod(ClosedProd(), Geometric(0.5), Geometric(0.6)) == Geometric(0.8)
        @test prod(ClosedProd(), Geometric(0.3), Geometric(0.8)) == Geometric(0.8600000000000001)
        @test prod(ClosedProd(), Geometric(0.5), Geometric(0.5)) == Geometric(0.75)

        η1 = ExponentialFamilyDistribution(Geometric, [log(1 - 0.6)])
        η2 = ExponentialFamilyDistribution(Geometric, [log(1 - 0.3)])
        @test prod(η1, η2) == ExponentialFamilyDistribution(Geometric, [log(0.4) + log(0.7)])
    end

    @testset "natural parameters related" begin
        d1 = Geometric(0.6)
        d2 = Geometric(0.3)
        η1 = ExponentialFamilyDistribution(Geometric, [log(1 - 0.6)])
        η2 = ExponentialFamilyDistribution(Geometric, [log(1 - 0.3)])

        @test convert(Geometric, η1) ≈ d1
        @test convert(Geometric, η2) ≈ d2

        @test convert(ExponentialFamilyDistribution, d1) == η1
        @test convert(ExponentialFamilyDistribution, d2) == η2

        @test logpartition(η1) ≈ -log(0.6)
        @test logpartition(η2) ≈ -log(0.3)

        @test basemeasure(η1, 4) == 1.0
        @test basemeasure(η2, 2) == 1.0

        @test logpdf(η1, 3) == logpdf(d1, 3)
        @test logpdf(η2, 3) == logpdf(d2, 3)

        @test pdf(η1, 3) == pdf(d1, 3)
        @test pdf(η2, 3) == pdf(d2, 3)

        @test isproper(ExponentialFamilyDistribution(Geometric, [log(0.6)])) == true
        @test isproper(ExponentialFamilyDistribution(Geometric, [1.3])) == false
    end

    @testset "fisher information" begin
        rng = StableRNG(42)
        n_samples = 10000

        transformation(η) = one(Float64) - exp(η[1])
        for p in 0.1:0.05:0.9
            dist = Geometric(p)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Geometric, η))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test first(fisherinformation(ef)) ≈ first(autograd_information(η)) atol = 1e-8

            J = ForwardDiff.gradient(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ first(fisherinformation(ef)) atol = 1e-8
        end
    end

    @testset "ExponentialFamilyDistribution mean,var" begin
        for p in 0.1:0.05:0.9
            dist = Geometric(p)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end
