module LaplaceTest

using Test
using ExponentialFamily
using Distributions
using Random
using DomainSets

import ExponentialFamily: mirrorlog, ExponentialFamilyDistribution, KnownExponentialFamilyDistribution, logpartition,
    basemeasure, getbasemeasure, getnaturalparameters, getsufficientstatistics, getconditioner

@testset "Laplace" begin
    @testset "vague" begin
        d = vague(Laplace)

        @test typeof(d) <: Laplace
        @test mean(d) === 0.0
        @test params(d) === (0.0, 1e12)
    end

    @testset "prod" begin
        l_left = Laplace(1.0, 0.2)
        l_right = Laplace(1.0, 0.1)
        @test prod(ClosedProd(), l_left, l_right) == Laplace(1.0, 1 / 15)

        l_right2 = Laplace(2.0, 0.6)
        ef_left = convert(KnownExponentialFamilyDistribution, l_left)
        ef_right = convert(KnownExponentialFamilyDistribution, l_right2)
        (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
        (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
        basemeasure = (x) -> 1.0
        sufficientstatistics = (x) -> [abs(x - conditioner_left), abs(x - conditioner_right)]
        function logpartition(η)
            A = sum(η)
            B = exp(sum(η))
            return -η[1] * conditioner_left - η[2] * conditioner_right - log(A) +
                   log(abs(B^conditioner_left - B^conditioner_right))
        end
        naturalparameters = [η_left, η_right]
        supp = support(l_left)
        @test getnaturalparameters(prod(ClosedProd(), l_left, l_right2)) == naturalparameters
        @test support(prod(ClosedProd(), l_left, l_right2)) == supp
        @test getbasemeasure(prod(ClosedProd(), l_left, l_right2))(1.0) == basemeasure(1.0)
        @test getsufficientstatistics(prod(ClosedProd(), l_left, l_right2))(1.0) ==
              sufficientstatistics(1.0)
    end

    @testset "LaplaceKnownExponentialFamilyDistribution" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(Distribution, KnownExponentialFamilyDistribution(Laplace, [-i], 2.0)) ==
                      Laplace(2.0, 1 / i)

                @test convert(KnownExponentialFamilyDistribution, Laplace(sqrt(i), i)) ==
                      KnownExponentialFamilyDistribution(Laplace, [-1 / i], sqrt(i))
            end
        end

        @testset "logpartition" begin
            @test logpartition(KnownExponentialFamilyDistribution(Laplace, [-1.0], 1.0)) ≈ log(2)
            @test logpartition(KnownExponentialFamilyDistribution(Laplace, [-2.0], 1.0)) ≈ log(4)
        end

        @testset "logpdf" begin
            for i in 1:10
                @test logpdf(KnownExponentialFamilyDistribution(Laplace, [-i], 0.0), 0.01) ≈
                      logpdf(Laplace(0.0, 1 / i), 0.01)
                @test logpdf(KnownExponentialFamilyDistribution(Laplace, [-i], 1.0), 0.5) ≈
                      logpdf(Laplace(1.0, 1 / i), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(Laplace, [-i], 1.0)) === true
                @test isproper(KnownExponentialFamilyDistribution(Laplace, [i], 2.0)) === false
            end
        end

        @testset "basemeasure" begin
            for (i) in (1:10)
                @test basemeasure(KnownExponentialFamilyDistribution(Laplace, [-i], 1.0), i^2) == 1.0
            end
        end
    end
end

end
