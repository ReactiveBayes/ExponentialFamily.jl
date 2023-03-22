module VonMisesTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, logpartition, basemeasure
import SpecialFunctions: besselj0
@testset "VonMises" begin

    # VonMises comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(VonMises)

        @test typeof(d) <: VonMises
        @test mean(d) === 0.0
        @test params(d) === (0.0, 1.0e-12)
    end

    @testset "prod" begin
        @test prod(ClosedProd(), VonMises(3.0, 2.0), VonMises(2.0, 1.0)) ≈ Base.convert(
            Distribution,
            prod(
                convert(KnownExponentialFamilyDistribution, VonMises(3.0, 2.0)),
                convert(KnownExponentialFamilyDistribution, VonMises(2.0, 1.0))
            )
        )
        @test prod(ClosedProd(), VonMises(7.0, 1.0), VonMises(0.1, 4.5)) ≈ Base.convert(
            Distribution,
            prod(
                convert(KnownExponentialFamilyDistribution, VonMises(7.0, 1.0)),
                convert(KnownExponentialFamilyDistribution, VonMises(0.1, 4.5))
            )
        )
        @test prod(ClosedProd(), VonMises(1.0, 3.0), VonMises(0.2, 0.4)) ≈ Base.convert(
            Distribution,
            prod(
                convert(KnownExponentialFamilyDistribution, VonMises(1.0, 3.0)),
                convert(KnownExponentialFamilyDistribution, VonMises(0.2, 0.4))
            )
        )
    end

    @testset "VonMisesKnownExponentialFamilyDistribution" begin
        @testset "Constructor" begin
            for i in 1:10, j in 1:10
                @test convert(Distribution, KnownExponentialFamilyDistribution(VonMises, [i, j])) ==
                      VonMises(acos(i / sqrt(i^2 + j^2)), sqrt(i^2 + j^2))

                @test convert(
                    KnownExponentialFamilyDistribution,
                    VonMises(acos(i / sqrt(i^2 + j^2)), sqrt(i^2 + j^2))
                ) ≈
                      KnownExponentialFamilyDistribution(VonMises, float([i, j]))
            end
        end

        @testset "logpartition" begin
            @test logpartition(KnownExponentialFamilyDistribution(VonMises, [2 / √(2), 2 / √(2)])) ≈ log(besselj0(2))
            @test logpartition(KnownExponentialFamilyDistribution(VonMises, [1, 1])) ≈ log(besselj0(sqrt(2)))
        end

        @testset "logpdf" begin
            for i in 1:10, j in 1:10
                @test logpdf(KnownExponentialFamilyDistribution(VonMises, [i, j]), 0.01) ≈
                      logpdf(VonMises(acos(i / sqrt(i^2 + j^2)), sqrt(i^2 + j^2)), 0.01)
                @test logpdf(KnownExponentialFamilyDistribution(VonMises, [i, j]), 0.5) ≈
                      logpdf(VonMises(acos(i / sqrt(i^2 + j^2)), sqrt(i^2 + j^2)), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 0:10
                @test isproper(KnownExponentialFamilyDistribution(VonMises, [i, i])) === true
            end
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(VonMises, [-i, -i])) === true
            end
        end

        @testset "basemeasure" begin
            for (i, j) in (1:10, 1:10)
                @test basemeasure(KnownExponentialFamilyDistribution(VonMises, [i, j]), rand()) == 1 / 2pi
                @test basemeasure(VonMises(i + 1, j + 1), rand()) == 1 / 2pi
            end
        end
    end
end

end
