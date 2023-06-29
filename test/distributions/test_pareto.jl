module ParetoTest

using Test
using Random
using Distributions
using ExponentialFamily
using ForwardDiff
using StableRNGs
import ExponentialFamily:
    KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation, getconditioner

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
        @test isproper(KnownExponentialFamilyDistribution(Pareto, -2.0, 1)) == true
        @test_throws AssertionError KnownExponentialFamilyDistribution(Pareto, -2.0, 2.1)
        @test_throws MethodError KnownExponentialFamilyDistribution(Pareto, 1.3)
    end

    @testset "prod" begin
        @test prod(ClosedProd(), Pareto(0.5), Pareto(0.6)) == Pareto(2.1)
        @test prod(ClosedProd(), Pareto(0.3), Pareto(0.8)) == Pareto(2.1)
        @test prod(ClosedProd(), Pareto(0.5), Pareto(0.5)) == Pareto(2.0)
        @test prod(ClosedProd(), Pareto(3), Pareto(2)) == Pareto(6.0)
    end

    @testset "natural parameters related" begin
        @test Distributions.logpdf(Pareto(10.0, 1.0), 1.0) ≈
              Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Pareto(10.0, 1.0)), 1.0)
        @test Distributions.logpdf(Pareto(5.0, 1.0), 1.0) ≈
              Distributions.logpdf(convert(KnownExponentialFamilyDistribution, Pareto(5.0, 1.0)), 1.0)
    end

    @testset "fisher information" begin
        rng = StableRNG(42)
        n_samples = 1000
        for λ in 1:10, u in 1:10
            dist = Pareto(λ, u)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            samples = rand(rng, Pareto(λ, u), n_samples)
            transformation(η) = [-1 - η, getconditioner(ef)]
            J = ForwardDiff.derivative(transformation, η)
            totalHessian = zeros(2, 2)
            for sample in samples
                totalHessian -= ForwardDiff.hessian((params) -> logpdf.(Pareto(params[1], params[2]), sample), [λ, u])
            end
            @test fisherinformation(dist) ≈ totalHessian / n_samples atol = 1e-8
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef)
            f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(Pareto, η, getconditioner(ef)))
            df = (η) -> ForwardDiff.derivative(f_logpartition, η)
            autograd_information = (η) -> ForwardDiff.derivative(df, η)
            @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-8
        end
    end

    @testset "KnownExponentialFamilyDistribution mean, var" begin
        for λ in 1:10, u in 1:10
            dist = Pareto(λ, u)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end
