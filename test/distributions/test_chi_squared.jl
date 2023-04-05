module ChisqTest

using Test
using ExponentialFamily
using Random
using Distributions

import SpecialFunctions: logfactorial
import ExponentialFamily: xtlog, KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure

@testset "Chisq" begin
    @testset "ChisqKnownExponentialFamilyDistribution" begin
        for i in 3:10
            @test convert(Distribution, KnownExponentialFamilyDistribution(Chisq, [i])) ≈ Chisq(2 * (i + 1))
            @test Distributions.logpdf(KnownExponentialFamilyDistribution(Chisq, [i]), 10) ≈
                  Distributions.logpdf(Chisq(2 * (i + 1)), 10)
            @test isproper(KnownExponentialFamilyDistribution(Chisq, [i])) === true
            @test isproper(KnownExponentialFamilyDistribution(Chisq, [-2 * i])) === false

            @test convert(KnownExponentialFamilyDistribution, Chisq(i)) ==
                  KnownExponentialFamilyDistribution(Chisq, [i / 2 - 1])
        end
    end

    @testset "prod" begin
        for i in 3:10
            left = Chisq(i + 1)
            right = Chisq(i)
            η_left = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, left)))
            η_right = first(getnaturalparameters(convert(KnownExponentialFamilyDistribution, right)))

            naturalparameters = [η_left + η_right - 2]
            basemeasure = (x) -> exp(-x)
            sufficientstatistics = (x) -> log(x)
            logpartition = (η) -> loggamma(η + 1)
            supp = support(left)
            @test getnaturalparameters(prod(ClosedProd(), left, right)) == naturalparameters
        end
    end

    @testset "Natural parameterization tests" begin
        @test Distributions.logpdf(Chisq(10), 1.0) ≈
              Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Chisq(10)), 1.0)
        @test Distributions.logpdf(Chisq(5), 1.0) ≈
              Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Chisq(5)), 1.0)
    end

    @test basemeasure(Chisq(5), 3) == exp(-3 / 2)
end

end
