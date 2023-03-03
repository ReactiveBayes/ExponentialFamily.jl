module ErlangTest

using Test
using ExponentialFamily
using Random
using Distributions

import SpecialFunctions: logfactorial
import ExponentialFamily: xtlog

@testset "Erlang" begin
    @testset "Constructor" begin

        @test Erlang() == Erlang(1, 1.0)
        @test Erlang(1.0, 2.0) == Erlang(1, 2.0)
        @test Erlang(1, 2) == Erlang(1, 2.0)
        @test Erlang(1.0, 2) == Erlang(1.0, 2.0)
        @test Erlang(1.0f0, 2.0f0) == Erlang(1, 2.0f0)
        @test Erlang(1.0f0, 2) == Erlang{Float32}(1, 2.0f0)
        @test Erlang(1.0f0, 2.0) == Erlang{Float64}(1, 2.0)

    end

    @testset "vague" begin
        vague(Erlang) == Erlang(1.0, 1e12)
    end

    @testset "ErlangNaturalParameters" begin
        for i in 2:10
            @test convert(Distribution, ErlangNaturalParameters(i, -i)) ≈ Erlang(i + 1, inv(i))
            @test Distributions.logpdf(ErlangNaturalParameters(i, -i), 10) ≈ Distributions.logpdf(Erlang(i + 1, inv(i)), 10)
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

    # TODO: lognormalizer etc
end

end
