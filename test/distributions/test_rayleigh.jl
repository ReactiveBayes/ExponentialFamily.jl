module RayleighTest

using Test
using ExponentialFamily
using Distributions
using Random

import ExponentialFamily: mirrorlog, NaturalParameters, get_params, lognormalizer, basemeasure

@testset "Rayleigh" begin

    # Rayleigh comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(Rayleigh)

        @test typeof(d) <: Rayleigh
        @test mean(d) ≈ d.σ * √(π / 2)
        @test params(d) === (1e12,)
    end

    # @testset "prod" begin
    #     @test prod(ProdAnalytical(), Rayleigh(3.0, 2.0), Rayleigh(2.0, 1.0)) ≈ Rayleigh(4.0, 2.0)
    #     @test prod(ProdAnalytical(), Rayleigh(7.0, 1.0), Rayleigh(0.1, 4.5)) ≈ Rayleigh(6.1, 4.5)
    #     @test prod(ProdAnalytical(), Rayleigh(1.0, 3.0), Rayleigh(0.2, 0.4)) ≈ Rayleigh(0.19999999999999996, 2.4)
    # end

    @testset "RayleighNaturalParameters" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(Distribution, NaturalParameters(Rayleigh, [-i])) == Rayleigh(sqrt(1 / 2i))

                @test convert(NaturalParameters, Rayleigh(i)) == NaturalParameters(Rayleigh, [-1 / (2i^2)])
            end
        end

        @testset "lognormalizer" begin
            @test lognormalizer(NaturalParameters(Rayleigh, -1.0)) ≈ log(2)
            @test lognormalizer(NaturalParameters(Rayleigh, -2.0)) ≈ log(4)
        end

        @testset "logpdf" begin
            for i in 1:10
                @test logpdf(NaturalParameters(Rayleigh, [-i]), 0.01) ≈ logpdf(Rayleigh(sqrt(1 / 2i)), 0.01)
                @test logpdf(NaturalParameters(Rayleigh, [-i]), 0.5) ≈ logpdf(Rayleigh(sqrt(1 / 2i)), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(NaturalParameters(Rayleigh, [-i])) === true
                @test isproper(NaturalParameters(Rayleigh, [i])) === false
            end
        end

        @testset "basemeasure" begin
            for (i) in (1:10)
                @test basemeasure(NaturalParameters(Rayleigh, [-i]), i^2) == i^2
                @test basemeasure(Rayleigh(i), i / 2) == i / 2
            end
        end

        @testset "+(::NaturalParameters{Rayleigh}, ::NaturalParameters{Rayleigh})" begin
            left = convert(NaturalParameters, Rayleigh(2))
            right = convert(NaturalParameters, Rayleigh(5))
            @test (left + right) == NaturalParameters(Rayleigh, [-1 / (2 * 2^2) - 1 / (2 * 5^2)])
            @test (left - right) == NaturalParameters(Rayleigh, [-1 / (2 * 2^2) + 1 / (2 * 5^2)])
        end
    end
end

end
