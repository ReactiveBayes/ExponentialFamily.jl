module ErlangTest

using Test
using ExponentialFamily
using Random
using Distributions

import SpecialFunctions: logfactorial
import ExponentialFamily: xtlog, KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure

@testset "Erlang" begin
    @testset "Constructors" begin
        @test Erlang() == Erlang(1, 1.0)
        @test vague(Erlang) == Erlang(1, 1e12)
    end

    @testset "ErlangKnownExponentialFamilyDistribution" begin
        for i in 2:10
            @test convert(Distribution, KnownExponentialFamilyDistribution(Erlang, [i, -i])) ≈ Erlang(i + 1, inv(i))
            @test Distributions.logpdf(KnownExponentialFamilyDistribution(Erlang, [i, -i]), 10) ≈
                  Distributions.logpdf(Erlang(i + 1, inv(i)), 10)
            @test isproper(KnownExponentialFamilyDistribution(Erlang, [i, -i])) === true
            @test isproper(KnownExponentialFamilyDistribution(Erlang, [-i, i])) === false

            @test convert(KnownExponentialFamilyDistribution, Erlang(i + 1, inv(i))) ==
                  KnownExponentialFamilyDistribution(Erlang, [i, -i])
        end
    end

    @testset "prod" begin
        @test prod(ClosedProd(), Erlang(1, 1), Erlang(1, 1)) == Erlang(1, 1 / 2)
        @test prod(ClosedProd(), Erlang(1, 2), Erlang(1, 1)) == Erlang(1, 2 / 3)
        @test prod(ClosedProd(), Erlang(1, 2), Erlang(1, 2)) == Erlang(1, 1)
        @test prod(ClosedProd(), Erlang(2, 2), Erlang(1, 2)) == Erlang(2, 1)
        @test prod(ClosedProd(), Erlang(2, 2), Erlang(2, 2)) == Erlang(3, 1)
    end

    @testset "Base operations" begin
        @test prod(
            KnownExponentialFamilyDistribution(Erlang, [4, 2.0]),
            KnownExponentialFamilyDistribution(Erlang, [2, 3.0])
        ) ==
              KnownExponentialFamilyDistribution(Erlang, [6, 5.0])
    end

    @testset "Natural parameterization tests" begin
        @test Distributions.logpdf(Erlang(10, 4.0), 1.0) ≈
              Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Erlang(10, 4.0)), 1.0)
        @test Distributions.logpdf(Erlang(5, 2.0), 1.0) ≈
              Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Erlang(5, 2.0)), 1.0)
    end

    @test basemeasure(Erlang(5, 2.0), rand(3)) == 1.0
end

end
