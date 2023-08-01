module ParetoTest

using Test
using Random
using Distributions
using ExponentialFamily
using ForwardDiff
using StableRNGs
import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation, getconditioner

@testset "Pareto" begin
    @testset "Stats methods" begin
        d = Pareto(3.0)
        @test Pareto() == Pareto(1.0)
        @test typeof(vague(Pareto)) <: Pareto
        @test vague(Pareto) == Pareto(1e12)
        @test shape(d) == 3.0
        @test scale(d) == 1.0
        @test mean(d) == 1.5
        @test var(d) == 0.75
    end

    @testset "isproper" begin
        @test isproper(ExponentialFamilyDistribution(Pareto, [-2.0], 1)) == true
        @test_throws AssertionError ExponentialFamilyDistribution(Pareto, [-2.0], 2.1)
        @test_throws MethodError ExponentialFamilyDistribution(Pareto, [1.3])
    end

    @testset "prod" begin
        @test prod(ClosedProd(), Pareto(0.5), Pareto(0.6)) == Pareto(2.1)
        @test prod(ClosedProd(), Pareto(0.3), Pareto(0.8)) == Pareto(2.1)
        @test prod(ClosedProd(), Pareto(0.5), Pareto(0.5)) == Pareto(2.0)
        @test prod(ClosedProd(), Pareto(3), Pareto(2)) == Pareto(6.0)
    end

    @testset "natural parameters related" begin
        @test Distributions.logpdf(Pareto(10.0, 1.0), 1.0) ≈
              Distributions.logpdf(convert(ExponentialFamilyDistribution, Pareto(10.0, 1.0)), 1.0)
        @test Distributions.logpdf(Pareto(5.0, 1.0), 1.0) ≈
              Distributions.logpdf(convert(ExponentialFamilyDistribution, Pareto(5.0, 1.0)), 1.0)
    end

    @testset "fisher information" begin
        rng = StableRNG(42)
        n_samples = 1000
        for λ in 1:10, u in 1:10
            dist = Pareto(λ, u)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            samples = rand(rng, Pareto(λ, u), n_samples)
            transformation(η) = [-1 - η[1], getconditioner(ef)]
            J = ForwardDiff.jacobian(transformation, η)
            totalHessian = zeros(2, 2)
            # for sample in samples
            #     totalHessian -= ForwardDiff.hessian((params) -> logpdf.(Pareto(params[1], params[2]), sample), [λ, u])
            # end
            # @test fisherinformation(dist) ≈ totalHessian / n_samples atol = 1e-8
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef)
            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Pareto, η, getconditioner(ef)))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-8
        end
    end

    @testset "ExponentialFamilyDistribution mean, var" begin
        for λ in 1:10, u in 1:10
            dist = Pareto(λ, u)
            ef = convert(ExponentialFamilyDistribution, dist)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end
