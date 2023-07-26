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
    xtlog, ExponentialFamilyDistribution, getnaturalparameters, basemeasure, ExponentialFamilyDistribution,
    fisherinformation, sufficientstatistics

@testset "Chisq" begin
    @testset "naturalparameters" begin
        for i in 3:10
            @test convert(Distribution, ExponentialFamilyDistribution(Chisq, [i])) ≈ Chisq(2 * (i + 1))
            @test Distributions.logpdf(ExponentialFamilyDistribution(Chisq, [i]), 10) ≈
                  Distributions.logpdf(Chisq(2 * (i + 1)), 10)
            @test isproper(ExponentialFamilyDistribution(Chisq, [i])) === true
            @test isproper(ExponentialFamilyDistribution(Chisq, [-2 * i])) === false

            @test convert(ExponentialFamilyDistribution, Chisq(i)) ==
                  ExponentialFamilyDistribution(Chisq, [i / 2 - 1])

            @test Distributions.logpdf(Chisq(10), 1.0) ≈
                  Distributions.logpdf(convert(ExponentialFamilyDistribution, Chisq(10)), 1.0)
            @test Distributions.logpdf(Chisq(5), 1.0) ≈
                  Distributions.logpdf(convert(ExponentialFamilyDistribution, Chisq(5)), 1.0)
        end

        chisqef = ExponentialFamilyDistribution(Chisq, [3])
        @test sufficientstatistics(chisqef, 1) == [log(1)]
        @test_throws AssertionError sufficientstatistics(chisqef, -1)
    end

    @testset "fisherinformation ExponentialFamilyDistribution{Chisq}" begin
        f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Chisq, η))
        autograd_inforamation_matrix = (η) -> ForwardDiff.hessian(f_logpartition, η)
        for i in 3:10
            @test first(fisherinformation(ExponentialFamilyDistribution(Chisq, [i]))) ≈
                  first(autograd_inforamation_matrix([i]))
        end
    end

    transformation(η) = 2 * (η[1] + 1)

    @testset "fisherinformation (Chisq)" begin
        rng = StableRNG(42)
        n_samples = 1000
        for ν in 1:10
            dist = Chisq(ν)
            ef = convert(ExponentialFamilyDistribution, Chisq(ν))
            chisq_fisher = fisherinformation(dist)
            ef_fisher = fisherinformation(ef)
            η = getnaturalparameters(ef)
            J = ForwardDiff.gradient(transformation, η)
            @test J' * chisq_fisher * J ≈ ef_fisher atol = 0.01
        end
    end

    @testset "prod" begin
        for i in 3:20
            left = Chisq(i + 1)
            right = Chisq(i)
            efleft = convert(ExponentialFamilyDistribution, left)
            efright = convert(ExponentialFamilyDistribution, right)
            prod_dist = prod(ClosedProd(), left, right)
            prod_ef = prod(efleft, efright)
            η_left = getnaturalparameters(efleft)
            η_right = getnaturalparameters(efright)
            naturalparameters = η_left + η_right

            @test prod_dist.naturalparameters == naturalparameters
            @test prod_dist.basemeasure(i) ≈ exp(-i)
            @test prod_dist.sufficientstatistics(i) ≈ [log(i)]
            @test prod_dist.logpartition(η_left + η_right) ≈ loggamma(η_left[1] + η_right[1] + 1)
            @test prod_dist.support === support(left)

            @test prod_ef.naturalparameters == naturalparameters
            @test prod_ef.basemeasure(i) ≈ exp(-i)
            @test prod_ef.sufficientstatistics(i) ≈ [log(i)]
            @test prod_ef.logpartition(η_left + η_right) ≈ loggamma(η_left[1] + η_right[1] + 1)
            @test prod_ef.support === support(left)
        end
    end

    @testset "ExponentialFamilyDistribution mean var" begin
        for ν in 1:10
            dist = Chisq(ν)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end

end
