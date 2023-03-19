module VonMisesFisherTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, lognormalizer, basemeasure
import SpecialFunctions: besselj
@testset "VonMisesFisher" begin

    # VonMisesFisher comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(VonMisesFisher, 3)

        @test typeof(d) <: VonMisesFisher
        @test mean(d) == zeros(3)
        @test params(d) == (zeros(3), 1.0e-12)
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), VonMisesFisher([sin(30), cos(30)], 3.0), VonMisesFisher([sin(45), cos(45)], 4.0)) ≈
              Base.convert(
            Distribution,
            convert(ExponentialFamilyDistribution, VonMisesFisher([sin(30), cos(30)], 3.0)) +
            convert(ExponentialFamilyDistribution, VonMisesFisher([sin(45), cos(45)], 4.0))
        )
        @test prod(ProdAnalytical(), VonMisesFisher([sin(15), cos(15)], 5.0), VonMisesFisher([cos(20), sin(20)], 2.0)) ≈
              Base.convert(
            Distribution,
            convert(ExponentialFamilyDistribution, VonMisesFisher([sin(15), cos(15)], 5.0)) +
            convert(ExponentialFamilyDistribution, VonMisesFisher([cos(20), sin(20)], 2.0))
        )
    end

    @testset "VonMisesFisherExponentialFamilyDistribution" begin
        @testset "Constructor" begin
            for i in 1:6:360
                @test convert(Distribution, ExponentialFamilyDistribution(VonMisesFisher, 3 * [cos(i), sin(i)])) ≈
                      VonMisesFisher([cos(i), sin(i)], 3)

                @test convert(ExponentialFamilyDistribution, VonMisesFisher([cos(i), sin(i)], 3)) ≈
                      ExponentialFamilyDistribution(VonMisesFisher, 3 * [cos(i), sin(i)])
            end
        end

        @testset "lognormalizer" begin
            @test lognormalizer(ExponentialFamilyDistribution(VonMisesFisher, 2 * [sin(15), cos(15)])) ≈ log(besselj(0, 2.0))
            @test lognormalizer(ExponentialFamilyDistribution(VonMisesFisher, 6 * [sin(25), cos(25)])) ≈ log(besselj(0, 6.0))
        end

        @testset "logpdf" begin
            for i in 1:10, j in 1:10
                @test logpdf(ExponentialFamilyDistribution(VonMisesFisher, 3 * [cos(i), sin(i)]), [0.01, 0.5]) ≈
                      logpdf(VonMisesFisher([cos(i), sin(i)], 3), [0.01, 0.5])
                @test logpdf(ExponentialFamilyDistribution(VonMisesFisher, 3 * [cos(2 * i), sin(2 * i)]), [0.5, 0.2]) ≈
                      logpdf(VonMisesFisher([cos(2 * i), sin(2 * i)], 3), [0.5, 0.2])
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(ExponentialFamilyDistribution(VonMisesFisher, [i, i])) === true
            end
            for i in 1:10
                @test isproper(ExponentialFamilyDistribution(VonMisesFisher, [-i, -i])) === false
            end
            @test isproper(ExponentialFamilyDistribution(VonMisesFisher, [0, 0])) === true
        end

        @testset "basemeasure" begin
            for (i, j) in (1:10, 1:10)
                @test basemeasure(ExponentialFamilyDistribution(VonMisesFisher, [i, j]), rand(2)) == 1 / 2pi
                @test basemeasure(VonMisesFisher([sin(i), cos(i)],), rand(2)) == 1 / 2pi
            end
        end
    end
end

end
