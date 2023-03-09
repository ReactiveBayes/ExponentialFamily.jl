module ErlangTest

using Test
using ExponentialFamily
using Random
using Distributions

import SpecialFunctions: logfactorial
import ExponentialFamily: xtlog

@testset "Erlang" begin
    @testset "Constructors" begin
        @test Erlang() == Erlang(1, 1.0)
        @test vague(Erlang) == Erlang(1, 1e12)
    end

    @testset "ErlangNaturalParameters" begin
        for i in 2:10
            @test convert(Distribution, ErlangNaturalParameters(i, -i)) ≈ Erlang(i + 1, inv(i))
            @test Distributions.logpdf(ErlangNaturalParameters(i, -i), 10) ≈
                  Distributions.logpdf(Erlang(i + 1, inv(i)), 10)
            @test isproper(ErlangNaturalParameters(i, -i)) === true
            @test isproper(ErlangNaturalParameters(-i, i)) === false

            @test convert(ErlangNaturalParameters, i, -i) == ErlangNaturalParameters(i, -i)

            @test as_naturalparams(ErlangNaturalParameters, i, -i) == ErlangNaturalParameters(i, -i)
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
        @test ErlangNaturalParameters(1.0, 2.0) - ErlangNaturalParameters(2.0, 3.0) == ErlangNaturalParameters(-1, -1.0)
        @test ErlangNaturalParameters(4, 2.0) + ErlangNaturalParameters(2.0, 3.0) == ErlangNaturalParameters(6, 5.0)
    end

    @testset "Natural parameterization tests" begin
        @test naturalparams(Erlang(2, 5.0)) == ErlangNaturalParameters(1, -1 / 5.0)
        @test naturalparams(Erlang(3, 10)) == ErlangNaturalParameters(2, -0.1)
        @test convert(Erlang, ErlangNaturalParameters(1.0, -0.2)) == Erlang(2, 5.0)
        @test logpdf(Erlang(10, 4.0), 1.0) ≈ logpdf(naturalparams(Erlang(10, 4.0)), 1.0)
        @test logpdf(Erlang(5, 2.0), 1.0) ≈ logpdf(naturalparams(Erlang(5, 2.0)), 1.0)
    end
end

end
