module ParetoTest

using Test
using Random
using Distributions
using ExponentialFamily
import ExponentialFamily: NaturalParameters, get_params, basemeasure

@testset "Pareto" begin
    @testset "Pareto vague" begin
        d = Pareto(3.0)
        @test Pareto() == Pareto(1.0)
        @test typeof(vague(Pareto)) <: Pareto
        @test vague(Pareto) == Pareto(1e12)
        @test shape(d) == 3.0
        @test scale(d) == 1.0
        @test mean(d) == 1.5
        @test var(d) == 0.75
    end
    @testset "Pareto prod" begin
        @test prod(ProdAnalytical(), Pareto(0.5), Pareto(0.6)) == Pareto(2.1)
        @test prod(ProdAnalytical(), Pareto(0.3), Pareto(0.8)) == Pareto(2.1)
        @test prod(ProdAnalytical(), Pareto(0.5), Pareto(0.5)) == Pareto(2.0)
        @test prod(ProdAnalytical(), Pareto(3), Pareto(2)) == Pareto(6.0)
    end

    @testset "Natural parameterization related Pareto" begin
        @test Distributions.logpdf(Pareto(10.0, 1.0), 1.0) ≈
              Distributions.logpdf(convert(NaturalParameters, Pareto(10.0, 1.0)), 1.0)
        @test Distributions.logpdf(Pareto(5.0, 1.0), 1.0) ≈
              Distributions.logpdf(convert(NaturalParameters, Pareto(5.0, 1.0)), 1.0)
    end
    @testset "isproper" begin
        @test isproper(NaturalParameters(Pareto, [-2.0], 1)) == true
        @test_throws AssertionError NaturalParameters(Pareto, [-2.0], 2.1)
        @test_throws MethodError isproper(NaturalParameters(Pareto, [1.3]))
    end
end
end
