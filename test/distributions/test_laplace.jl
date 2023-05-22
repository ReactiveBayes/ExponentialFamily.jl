module LaplaceTest

using Test
using ExponentialFamily
using Distributions
using Random
using StableRNGs
using Zygote
using ForwardDiff

import ExponentialFamily: mirrorlog, ExponentialFamilyDistribution, KnownExponentialFamilyDistribution, logpartition,
    basemeasure, getbasemeasure, getnaturalparameters, getsufficientstatistics, getconditioner, fisherinformation,
    logpdf

@testset "Laplace" begin
    @testset "vague" begin
        d = vague(Laplace)

        @test typeof(d) <: Laplace
        @test mean(d) === 0.0
        @test params(d) === (0.0, 1e12)
    end

    @testset "prod" begin
        l_left = Laplace(1.0, 0.2)
        l_right = Laplace(1.0, 0.1)
        @test prod(ClosedProd(), l_left, l_right) == Laplace(1.0, 1 / 15)

        l_right2 = Laplace(2.0, 0.6)
        ef_left = convert(KnownExponentialFamilyDistribution, l_left)
        ef_right = convert(KnownExponentialFamilyDistribution, l_right2)
        (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
        (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
        basemeasure = (x) -> 1.0
        sufficientstatistics = (x) -> [abs(x - conditioner_left), abs(x - conditioner_right)]
        sorted_conditioner = sort([conditioner_left, conditioner_right])
        function logpartition(η)
            A1 = exp(η[1] * conditioner_left + η[2] * conditioner_right)
            A2 = exp(-η[1] * conditioner_left + η[2] * conditioner_right)
            A3 = exp(-η[1] * conditioner_left - η[2] * conditioner_right)
            B1 = (exp(sorted_conditioner[2] * (-η[1] - η[2])) - 1.0) / (-η[1] - η[2])
            B2 =
                (exp(sorted_conditioner[1] * (η[1] - η[2])) - exp(sorted_conditioner[2] * (η[1] - η[2]))) /
                (η[1] - η[2])
            B3 = (1.0 - exp(sorted_conditioner[1] * (η[1] + η[2]))) / (η[1] + η[2])

            return log(A1 * B1 + A2 * B2 + A3 * B3)
        end
        naturalparameters = [η_left, η_right]
        supp = support(l_left)
        @test getnaturalparameters(prod(ClosedProd(), l_left, l_right2)) == naturalparameters
        @test support(prod(ClosedProd(), l_left, l_right2)) == supp
        @test getbasemeasure(prod(ClosedProd(), l_left, l_right2))(1.0) == basemeasure(1.0)
        @test getsufficientstatistics(prod(ClosedProd(), l_left, l_right2))(1.0) ==
              sufficientstatistics(1.0)
    end

    @testset "natural parameters related" begin
        @testset "convert" begin
            for i in 1:10
                @test convert(Distribution, KnownExponentialFamilyDistribution(Laplace, [-i], 2.0)) ==
                      Laplace(2.0, 1 / i)

                @test convert(KnownExponentialFamilyDistribution, Laplace(sqrt(i), i)) ==
                      KnownExponentialFamilyDistribution(Laplace, [-1 / i], sqrt(i))
            end
        end

        @testset "logpartition" begin
            @test logpartition(KnownExponentialFamilyDistribution(Laplace, [-1.0], 1.0)) ≈ log(2)
            @test logpartition(KnownExponentialFamilyDistribution(Laplace, [-2.0], 1.0)) ≈ log(1)
        end

        @testset "logpdf" begin
            for i in 1:10
                @test logpdf(KnownExponentialFamilyDistribution(Laplace, [-i], 0.0), 0.01) ≈
                      logpdf(Laplace(0.0, 1 / i), 0.01)
                @test logpdf(KnownExponentialFamilyDistribution(Laplace, [-i], 1.0), 0.5) ≈
                      logpdf(Laplace(1.0, 1 / i), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(Laplace, [-i], 1.0)) === true
                @test isproper(KnownExponentialFamilyDistribution(Laplace, [i], 2.0)) === false
            end
        end

        @testset "basemeasure" begin
            for (i) in (1:10)
                @test basemeasure(KnownExponentialFamilyDistribution(Laplace, [-i], 1.0), i^2) == 1.0
            end
        end
        @testset "fisher information" begin
            rng = StableRNG(42)
            n_samples = 1000
            for λ in 1:10, u in 1.0:0.5:5.0
                dist = Laplace(u, λ)
                ef = convert(KnownExponentialFamilyDistribution, dist)
                η = getnaturalparameters(ef)

                f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(Laplace, η, getconditioner(ef)))
                autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
                @test fisherinformation(ef) ≈ first(autograd_information(η)) atol = 1e-8
            end
        end
    end
end

end
