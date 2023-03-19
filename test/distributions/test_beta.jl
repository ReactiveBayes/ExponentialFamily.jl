module BetaTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: mirrorlog, ExponentialFamilyDistribution, getnaturalparameters, lognormalizer, basemeasure
import SpecialFunctions: loggamma

@testset "Beta" begin

    # Beta comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(Beta)

        @test typeof(d) <: Beta
        @test mean(d) === 0.5
        @test params(d) === (1.0, 1.0)
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Beta(3.0, 2.0), Beta(2.0, 1.0)) ≈ Beta(4.0, 2.0)
        @test prod(ProdAnalytical(), Beta(7.0, 1.0), Beta(0.1, 4.5)) ≈ Beta(6.1, 4.5)
        @test prod(ProdAnalytical(), Beta(1.0, 3.0), Beta(0.2, 0.4)) ≈ Beta(0.19999999999999996, 2.4)
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Beta(1.0, 3.0)) ≈ -1.8333333333333335
        @test mean(log, Beta(0.1, 0.3)) ≈ -7.862370395825961
        @test mean(log, Beta(4.5, 0.3)) ≈ -0.07197681436958758
    end

    @testset "mean(::typeof(mirrorlog))" begin
        @test mean(mirrorlog, Beta(1.0, 3.0)) ≈ -0.33333333333333337
        @test mean(mirrorlog, Beta(0.1, 0.3)) ≈ -0.9411396776150167
        @test mean(mirrorlog, Beta(4.5, 0.3)) ≈ -4.963371962929249
    end

    @testset "BetaExponentialFamilyDistribution" begin
        @testset "Constructor" begin
            for i in 0:10, j in 0:10
                @test convert(Distribution, ExponentialFamilyDistribution(Beta, [i, j])) == Beta(i + 1, j + 1)

                @test convert(ExponentialFamilyDistribution, Beta(i + 1, j + 1)) == ExponentialFamilyDistribution(Beta, [i, j])
            end
        end

        @testset "lognormalizer" begin
            @test lognormalizer(ExponentialFamilyDistribution(Beta, [0, 0])) ≈ 0
            @test lognormalizer(ExponentialFamilyDistribution(Beta, [1, 1])) ≈ -loggamma(4)
        end

        @testset "logpdf" begin
            for i in 0:10, j in 0:10
                @test logpdf(ExponentialFamilyDistribution(Beta, [i, j]), 0.01) ≈ logpdf(Beta(i + 1, j + 1), 0.01)
                @test logpdf(ExponentialFamilyDistribution(Beta, [i, j]), 0.5) ≈ logpdf(Beta(i + 1, j + 1), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 0:10
                @test isproper(ExponentialFamilyDistribution(Beta, [i, i])) === true
            end
            for i in 1:10
                @test isproper(ExponentialFamilyDistribution(Beta, [-i, -i])) === false
            end
        end

        @testset "basemeasure" begin
            for (i, j) in (1:10, 1:10)
                @test basemeasure(ExponentialFamilyDistribution(Beta, [i, j]), rand()) == 1.0
                @test basemeasure(Beta(i + 1, j + 1), rand()) == 1.0
            end
        end

        @testset "+(::ExponentialFamilyDistribution{Beta}, ::ExponentialFamilyDistribution{Beta})" begin
            left = convert(ExponentialFamilyDistribution, Beta(2))
            right = convert(ExponentialFamilyDistribution, Beta(5))
            @test (left + right) == convert(ExponentialFamilyDistribution, Beta(6))
        end
    end
end

end
