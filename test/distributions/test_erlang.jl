module ErlangTest

using Test
using ExponentialFamily
using Random
using Distributions

import SpecialFunctions: logfactorial
import ExponentialFamily: xtlog, ExponentialFamilyDistribution, getnaturalparameters, basemeasure

@testset "Erlang" begin
    @testset "Constructors" begin
        @test Erlang() == Erlang(1, 1.0)
        @test vague(Erlang) == Erlang(1, 1e12)
    end

    @testset "ErlangExponentialFamilyDistribution" begin
        for i in 2:10
            @test convert(Distribution, ExponentialFamilyDistribution(Erlang, [i, -i])) ≈ Erlang(i + 1, inv(i))
            @test Distributions.logpdf(ExponentialFamilyDistribution(Erlang, [i, -i]), 10) ≈
                  Distributions.logpdf(Erlang(i + 1, inv(i)), 10)
            @test isproper(ExponentialFamilyDistribution(Erlang, [i, -i])) === true
            @test isproper(ExponentialFamilyDistribution(Erlang, [-i, i])) === false

            @test convert(ExponentialFamilyDistribution, Erlang(i + 1, inv(i))) == ExponentialFamilyDistribution(Erlang, [i, -i])
        end
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Erlang(1, 1), Erlang(1, 1)) == Erlang(1, 1 / 2)
        @test prod(ProdAnalytical(), Erlang(1, 2), Erlang(1, 1)) == Erlang(1, 2 / 3)
        @test prod(ProdAnalytical(), Erlang(1, 2), Erlang(1, 2)) == Erlang(1, 1)
        @test prod(ProdAnalytical(), Erlang(2, 2), Erlang(1, 2)) == Erlang(2, 1)
        @test prod(ProdAnalytical(), Erlang(2, 2), Erlang(2, 2)) == Erlang(3, 1)
    end

    @testset "Base operations" begin
        @test ExponentialFamilyDistribution(Erlang, [1, 2.0]) - ExponentialFamilyDistribution(Erlang, [2, 3.0]) ==
              ExponentialFamilyDistribution(Erlang, [-1, -1.0])
        @test ExponentialFamilyDistribution(Erlang, [4, 2.0]) + ExponentialFamilyDistribution(Erlang, [2, 3.0]) ==
              ExponentialFamilyDistribution(Erlang, [6, 5.0])
    end

    @testset "Natural parameterization tests" begin
        @test Distributions.logpdf(Erlang(10, 4.0), 1.0) ≈
              Distributions.logpdf(convert(ExponentialFamilyDistribution, Erlang(10, 4.0)), 1.0)
        @test Distributions.logpdf(Erlang(5, 2.0), 1.0) ≈
              Distributions.logpdf(convert(ExponentialFamilyDistribution, Erlang(5, 2.0)), 1.0)
    end

    @test basemeasure(Erlang(5, 2.0), rand(3)) == 1.0
end

end
