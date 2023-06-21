module ErlangTest

using Test
using ExponentialFamily
using Random
using Distributions
using ForwardDiff
import SpecialFunctions: logfactorial, loggamma
import ExponentialFamily:
    xtlog, KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

@testset "Erlang" begin
    @testset "Constructors" begin
        @test Erlang() == Erlang(1, 1.0)
        @test vague(Erlang) == Erlang(1, 1e12)
    end

    @testset "natural parameters related" begin
        for i in 2:10
            @test convert(Distribution, KnownExponentialFamilyDistribution(Erlang, [i, -i])) ≈ Erlang(i + 1, inv(i))
            @test Distributions.logpdf(KnownExponentialFamilyDistribution(Erlang, [i, -i]), 10) ≈
                  Distributions.logpdf(Erlang(i + 1, inv(i)), 10)
            @test isproper(KnownExponentialFamilyDistribution(Erlang, [i, -i])) === true
            @test isproper(KnownExponentialFamilyDistribution(Erlang, [-i, i])) === false

            @test convert(KnownExponentialFamilyDistribution, Erlang(i + 1, i)) ==
                  KnownExponentialFamilyDistribution(Erlang, [i, -inv(i)])

            @test Distributions.logpdf(Erlang(10, 4.0), 1.0) ≈
                  Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Erlang(10, 4.0)), 1.0)
            @test Distributions.logpdf(Erlang(5, 2.0), 1.0) ≈
                  Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Erlang(5, 2.0)), 1.0)
        end
    end

    @testset "prod" begin
        @test prod(ClosedProd(), Erlang(1, 1), Erlang(1, 1)) == Erlang(1, 1 / 2)
        @test prod(ClosedProd(), Erlang(1, 2), Erlang(1, 1)) == Erlang(1, 2 / 3)
        @test prod(ClosedProd(), Erlang(1, 2), Erlang(1, 2)) == Erlang(1, 1)
        @test prod(ClosedProd(), Erlang(2, 2), Erlang(1, 2)) == Erlang(2, 1)
        @test prod(ClosedProd(), Erlang(2, 2), Erlang(2, 2)) == Erlang(3, 1)
        @test prod(
            KnownExponentialFamilyDistribution(Erlang, [4, 2.0]),
            KnownExponentialFamilyDistribution(Erlang, [2, 3.0])
        ) ==
              KnownExponentialFamilyDistribution(Erlang, [6, 5.0])
    end

    @testset "fisher information" begin
        ## these functions are for testing purposes only.
        function transformation(params)
            a = getindex(params, 1)
            b = getindex(params, 2)
            return [a + 1, -b]
        end

        function lp(exponentialfamily::KnownExponentialFamilyDistribution{Erlang})
            η = getnaturalparameters(exponentialfamily)
            a = first(η)
            b = getindex(η, 2)
            return loggamma(a) - (a + one(a)) * log(-b)
        end

        for μ in 3:20, κ in 2.0:0.1:10.0
            dist = Erlang(μ, κ)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> lp(KnownExponentialFamilyDistribution(Erlang, η))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test fisherinformation(ef) ≈ autograd_information(η) rtol = 1e-6
            J = ForwardDiff.jacobian(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) rtol = 1e-6
        end
    end
end

end
