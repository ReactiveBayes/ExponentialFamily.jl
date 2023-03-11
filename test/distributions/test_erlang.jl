module ErlangTest

using Test
using ExponentialFamily
using Random
using Distributions

import SpecialFunctions: logfactorial
import ExponentialFamily: xtlog, NaturalParameters, get_params

@testset "Erlang" begin
    @testset "Constructors" begin
        @test Erlang() == Erlang(1, 1.0)
        @test vague(Erlang) == Erlang(1, 1e12)
    end

    @testset "ErlangNaturalParameters" begin
        for i in 2:10
            @test convert(Distribution, NaturalParameters(Erlang,[i, -i])) ≈ Erlang(i + 1, inv(i))
            @test Distributions.logpdf(NaturalParameters(Erlang,[i, -i]), 10) ≈
                  Distributions.logpdf(Erlang(i + 1, inv(i)), 10)
            @test isproper(NaturalParameters(Erlang,[i, -i])) === true
            @test isproper(NaturalParameters(Erlang,[-i, i])) === false

            @test convert(NaturalParameters, Erlang(i + 1, inv(i))) == NaturalParameters(Erlang,[i, -i])

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
        @test NaturalParameters(Erlang,[1, 2.0]) - NaturalParameters(Erlang,[2, 3.0]) == NaturalParameters(Erlang,[-1, -1.0])
        @test NaturalParameters(Erlang,[4, 2.0]) + NaturalParameters(Erlang,[2, 3.0]) == NaturalParameters(Erlang,[6, 5.0])
    end

    @testset "Natural parameterization tests" begin
        @test Distributions.logpdf(Erlang(10, 4.0), 1.0) ≈ Distributions.logpdf(convert(NaturalParameters,Erlang(10, 4.0)), 1.0)
        @test Distributions.logpdf(Erlang(5, 2.0), 1.0) ≈ Distributions.logpdf(convert(NaturalParameters,Erlang(5, 2.0)), 1.0)
    end
end

end
