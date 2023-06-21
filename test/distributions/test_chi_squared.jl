module ChisqTest

using Test
using ExponentialFamily
using Random
using Distributions
using ForwardDiff
using StableRNGs
using ForwardDiff

import SpecialFunctions: logfactorial, loggamma
import ExponentialFamily:
    xtlog, KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure, ExponentialFamilyDistribution,
    fisherinformation, sufficientstatistics

@testset "Chisq" begin
    @testset "naturalparameters" begin
        for i in 3:10
            @test convert(Distribution, KnownExponentialFamilyDistribution(Chisq, i)) ≈ Chisq(2 * (i + 1))
            @test Distributions.logpdf(KnownExponentialFamilyDistribution(Chisq, i), 10) ≈
                  Distributions.logpdf(Chisq(2 * (i + 1)), 10)
            @test isproper(KnownExponentialFamilyDistribution(Chisq, i)) === true
            @test isproper(KnownExponentialFamilyDistribution(Chisq, -2 * i)) === false

            @test convert(KnownExponentialFamilyDistribution, Chisq(i)) ==
                  KnownExponentialFamilyDistribution(Chisq, i / 2 - 1)

            @test Distributions.logpdf(Chisq(10), 1.0) ≈
                  Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Chisq(10)), 1.0)
            @test Distributions.logpdf(Chisq(5), 1.0) ≈
                  Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Chisq(5)), 1.0)
        end

        chisqef = KnownExponentialFamilyDistribution(Chisq, 3)
        @test sufficientstatistics(chisqef, 1) == log(1)
        @test_throws AssertionError sufficientstatistics(chisqef, -1)
    end

    @testset "fisherinformation KnownExponentialFamilyDistribution{Chisq}" begin
        f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(Chisq, η))
        df = (η) -> ForwardDiff.derivative(f_logpartition, η)
        autograd_inforamation_matrix = (η) -> ForwardDiff.derivative(df, η)
        for i in 3:10
            @test fisherinformation(KnownExponentialFamilyDistribution(Chisq, i)) ≈
                  autograd_inforamation_matrix(i)
        end
    end

    transformation(η) = 2 * (η + 1)

    @testset "fisherinformation (Chisq)" begin
        rng = StableRNG(42)
        n_samples = 1000
        for ν in 1:10
            dist = Chisq(ν)
            ef = convert(KnownExponentialFamilyDistribution, Chisq(ν))
            chisq_fisher = fisherinformation(dist)
            ef_fisher = fisherinformation(ef)
            η = getnaturalparameters(ef)
            J = ForwardDiff.derivative(transformation, η)
            @test chisq_fisher * J^2 ≈ ef_fisher atol = 0.01
        end
    end

    @testset "prod" begin
        for i in 3:20
            left = Chisq(i + 1)
            right = Chisq(i)
            efleft = convert(KnownExponentialFamilyDistribution, left)
            efright = convert(KnownExponentialFamilyDistribution, right)
            prod_dist = prod(ClosedProd(), left, right)
            prod_ef = prod(efleft, efright)

            η_left = first(getnaturalparameters(efleft))
            η_right = first(getnaturalparameters(efright))
            naturalparameters = η_left + η_right

            @test prod_dist.naturalparameters == naturalparameters
            @test prod_dist.basemeasure(i) ≈ exp(-i)
            @test prod_dist.sufficientstatistics(i) ≈ log(i)
            @test prod_dist.logpartition(η_left + η_right) ≈ loggamma(η_left + η_right + 1)
            @test prod_dist.support === support(left)

            @test prod_ef.naturalparameters == naturalparameters
            @test prod_ef.basemeasure(i) ≈ exp(-i)
            @test prod_ef.sufficientstatistics(i) ≈ log(i)
            @test prod_ef.logpartition(η_left + η_right) ≈ loggamma(η_left + η_right + 1)
            @test prod_ef.support === support(left)
        end
    end
end

end
