module RayleighTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff
using DomainSets
using StableRNGs

import ExponentialFamily: mirrorlog, ExponentialFamilyDistribution, KnownExponentialFamilyDistribution, logpartition,
    basemeasure, getbasemeasure, getnaturalparameters, getsufficientstatistics, fisherinformation

@testset "Rayleigh" begin
    @testset "vague" begin
        d = vague(Rayleigh)

        @test typeof(d) <: Rayleigh
        @test mean(d) ≈ d.σ * √(π / 2)
        @test params(d) === (1e12,)
    end

    @testset "prod" begin
        naturalparameters(σ1, σ2) = -0.5(σ1^2 + σ2^2) / (σ1 * σ2)^2
        basemeasure = (x) -> 4 * x^2 / sqrt(pi)
        sufficientstatistics = (x) -> x^2
        logpartition = (η) -> log(η^(-3 / 2))
        supp = DomainSets.HalfLine()
        @test getnaturalparameters(prod(ClosedProd(), Rayleigh(3.0), Rayleigh(2.0))) == naturalparameters(3.0, 2.0)
        @test support(prod(ClosedProd(), Rayleigh(7.0), Rayleigh(1.0))) == supp
        @test getbasemeasure(prod(ClosedProd(), Rayleigh(1.0), Rayleigh(2.0)))(1.0) == basemeasure(1.0)
        @test getsufficientstatistics(prod(ClosedProd(), Rayleigh(1.0), Rayleigh(2.0)))(1.0) ==
              sufficientstatistics(1.0)
    end

    @testset "natural parameters related" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(Distribution, KnownExponentialFamilyDistribution(Rayleigh, -i)) ==
                      Rayleigh(sqrt(1 / 2i))

                @test convert(KnownExponentialFamilyDistribution, Rayleigh(i)) ==
                      KnownExponentialFamilyDistribution(Rayleigh, -1 / (2i^2))
            end
        end

        @testset "logpartition" begin
            @test logpartition(KnownExponentialFamilyDistribution(Rayleigh, -1.0)) ≈ -log(2)
            @test logpartition(KnownExponentialFamilyDistribution(Rayleigh, -2.0)) ≈ -log(4)
        end

        @testset "logpdf" begin
            for i in 1:10
                @test logpdf(KnownExponentialFamilyDistribution(Rayleigh, -i), 0.01) ≈
                      logpdf(Rayleigh(sqrt(1 / 2i)), 0.01)
                @test logpdf(KnownExponentialFamilyDistribution(Rayleigh, -i), 0.5) ≈
                      logpdf(Rayleigh(sqrt(1 / 2i)), 0.5)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(Rayleigh, -i)) === true
                @test isproper(KnownExponentialFamilyDistribution(Rayleigh, i)) === false
            end
        end

        @testset "basemeasure" begin
            for (i) in (1:10)
                @test basemeasure(KnownExponentialFamilyDistribution(Rayleigh, -i), i^2) == i^2
            end
        end
        transformation(η) = sqrt(-1 / (2η))
        @testset "fisher information" begin
            rng = StableRNG(0)
            n_samples = 10000
            for λ in 1:10
                dist = Rayleigh(λ)
                ef = convert(KnownExponentialFamilyDistribution, dist)
                η = getnaturalparameters(ef)

                samples = rand(rng, Rayleigh(λ), n_samples)

                totalHessian = zeros(typeof(λ), 1, 1)
                for sample in samples
                    totalHessian -= ForwardDiff.hessian((params) -> logpdf.(Rayleigh(params[1]), sample), [λ])
                end
                @test fisherinformation(dist) ≈ first(totalHessian) / n_samples atol = 0.1

                f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(Rayleigh, η))
                df = (η) -> ForwardDiff.derivative(f_logpartition, η)
                autograd_information = (η) -> ForwardDiff.derivative(df, η)
                @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-8
                J = ForwardDiff.derivative(transformation, η)
                @test fisherinformation(dist) * J^2 ≈ fisherinformation(ef) atol = 1e-8
            end
        end
    end
end

end
