module PoissonTest

using Test
using ExponentialFamily
using Random
using Distributions
using ForwardDiff
using StableRNGs
using LinearAlgebra

import SpecialFunctions: logfactorial, besseli
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation
import DomainSets: NaturalNumbers

@testset "Poisson" begin
    @testset "convert" begin
        for i in 1:10
            @test convert(Distribution, ExponentialFamilyDistribution(Poisson, [log(i)])) ≈ Poisson(i)
            @test Distributions.logpdf(ExponentialFamilyDistribution(Poisson, [log(i)]), 10) ≈
                  Distributions.logpdf(Poisson(i), 10)

            @test convert(ExponentialFamilyDistribution, Poisson(i)) ==
                  ExponentialFamilyDistribution(Poisson, [log(i)])
        end
    end

    @testset "prod" begin
        left = Poisson(1)
        right = Poisson(1)
        prod_dist = prod(ClosedProd(), left, right)
        sample_points = collect(1:5)
        for x in sample_points
            @test prod_dist.basemeasure(x) == (1 / factorial(x)^2)
            @test prod_dist.sufficientstatistics(x) == [x]
        end
        sample_points = [-5, -2, 0, 2, 5]
        for η in sample_points
            @test prod_dist.logpartition(η) == log(abs(besseli(0, 2 * exp(η / 2))))
        end
        @test prod_dist.naturalparameters == log(1) + log(1)
        @test prod_dist.support == NaturalNumbers()

        sample_points = collect(1:5)
        for x in sample_points
            hist_sum(x) =
                prod_dist.basemeasure(x) * exp(
                    dot(prod_dist.sufficientstatistics(x) , prod_dist.naturalparameters) -
                    prod_dist.logpartition(prod_dist.naturalparameters)
                )
            @test sum(hist_sum(x) for x in 0:20) ≈ 1.0
        end
    end

    @testset "natural parameters related" begin
        @test Distributions.logpdf(Poisson(4), 1) ≈
              Distributions.logpdf(convert(ExponentialFamilyDistribution, Poisson(4)), 1)
        @test Distributions.logpdf(Poisson(5), 1) ≈
              Distributions.logpdf(convert(ExponentialFamilyDistribution, Poisson(5)), 1)

        for i in 2:10
            @test isproper(ExponentialFamilyDistribution(Poisson, [log(i)])) === true
            @test isproper(ExponentialFamilyDistribution(Poisson, [NaN])) === false
            @test isproper(ExponentialFamilyDistribution(Poisson, [Inf])) === false
        end
    end

    @testset "fisher information" begin
        for λ in 1:10
            dist = Poisson(λ)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)
            transformation(η) = exp(η[1])
            J = ForwardDiff.gradient(transformation, η)
            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Poisson, η))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-8
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) atol = 1e-8
        end
    end

    @testset "ExponentialFamilyDistribution mean, var" begin
        for λ in 1:10
            dist = Poisson(λ)
            ef = convert(ExponentialFamilyDistribution, dist)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end

end
