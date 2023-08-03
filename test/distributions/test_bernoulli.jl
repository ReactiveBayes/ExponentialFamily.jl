module BernoulliTest

using Test
using ExponentialFamily
using Distributions
using ForwardDiff
using Random
using StatsFuns
using StableRNGs

import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, compute_logscale, logpartition, basemeasure, insupport,
    sufficientstatistics, fisherinformation, pack_naturalparameters, unpack_naturalparameters, isbasemeasureconstant,
    ConstantBaseMeasure

@testset "Bernoulli" begin

    # Bernoulli comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(Bernoulli)

        @test typeof(d) <: Bernoulli
        @test mean(d) === 0.5
        @test succprob(d) === 0.5
        @test failprob(d) === 0.5
    end

    @testset "probvec" begin
        @test probvec(Bernoulli(0.5)) === (0.5, 0.5)
        @test probvec(Bernoulli(0.3)) === (0.7, 0.3)
        @test probvec(Bernoulli(0.6)) === (0.4, 0.6)
    end

    @testset "logscale Bernoulli-Bernoulli/Categorical" begin
        @test compute_logscale(Bernoulli(0.5), Bernoulli(0.5), Bernoulli(0.5)) ≈ log(0.5)
        @test compute_logscale(Bernoulli(1), Bernoulli(0.5), Bernoulli(1)) ≈ log(0.5)
        @test compute_logscale(Categorical([0.5, 0.5]), Bernoulli(0.5), Categorical([0.5, 0.5])) ≈ log(0.5)
        @test compute_logscale(Categorical([0.5, 0.5]), Categorical([0.5, 0.5]), Bernoulli(0.5)) ≈ log(0.5)
        @test compute_logscale(Categorical([1.0, 0.0]), Bernoulli(0.5), Categorical([1])) ≈ log(0.5)
        @test compute_logscale(Categorical([1.0, 0.0, 0.0]), Bernoulli(0.5), Categorical([1.0, 0, 0])) ≈ log(0.5)
    end

    @testset "ExponentialFamilyDistribution{Bernoulli}" begin

        # Check conversions and general statistics 
        @testset for p in 0.1:0.1:0.9
            @testset let d = Bernoulli(p)
                @test length(pack_naturalparameters(d)) === 1

                ef = convert(ExponentialFamilyDistribution, d)
                η₁ = log(p / (1 - p))

                @test all(unpack_naturalparameters(ef) .≈ (η₁,))
                @test @allocated(unpack_naturalparameters(ef)) === 0

                @test isproper(ef)
                @test ef isa ExponentialFamilyDistribution{Bernoulli}
                @test convert(Distribution, ef) ≈ d
                @test @allocated(convert(Distribution, ef)) === 0

                for x in (0, 1)
                    # We believe in the implementation in the `Distributions.jl`
                    @test logpdf(ef, x) ≈ logpdf(d, x)
                    @test pdf(ef, x) ≈ pdf(d, x)
                    @test mean(ef) ≈ mean(d)
                    @test var(ef) ≈ var(d)
                    @test std(ef) ≈ std(d)
                    @test rand(StableRNG(42), ef) ≈ rand(StableRNG(42), d)
                    @test all(rand(StableRNG(42), ef, 10) .≈ rand(StableRNG(42), d, 10))
                    @test all(rand!(StableRNG(42), ef, zeros(10)) .≈ rand!(StableRNG(42), d, zeros(10)))

                    @test isbasemeasureconstant(ef) === ConstantBaseMeasure()
                    @test basemeasure(ef, x) === one(x)
                    @test sufficientstatistics(ef, x) === (x,)
                    @test logpartition(ef) ≈ log(1 + exp(η₁))

                    # Test that the desired methods do not allocate
                    @test @allocated(basemeasure(ef, x)) === 0
                    @test @allocated(sufficientstatistics(ef, x)) === 0
                end

                @test !insupport(ef, -0.5)
                @test !insupport(ef, 0.5)

                # Not in the support
                @test_throws Exception logpdf(ef, 0.5)
                @test_throws Exception logpdf(ef, -0.5)
            end
        end
    end

    @testset "prod with Distribution" begin
        @test prod(ClosedProd(), Bernoulli(0.5), Bernoulli(0.5)) ≈ Bernoulli(0.5)
        @test prod(ClosedProd(), Bernoulli(0.1), Bernoulli(0.6)) ≈ Bernoulli(0.14285714285714285)
        @test prod(ClosedProd(), Bernoulli(0.78), Bernoulli(0.05)) ≈ Bernoulli(0.1572580645161291)
    end

    @testset "prod with ExponentialFamilyDistribution" for pleft in 0.1:0.8:0.9, pright in 0.1:0.8:0.9
        efleft = convert(ExponentialFamilyDistribution, Bernoulli(pleft))
        efright = convert(ExponentialFamilyDistribution, Bernoulli(pright))
        ηleft = getnaturalparameters(efleft)
        ηright = getnaturalparameters(efright)

        for strategy in (
            PreserveTypeProd(ExponentialFamilyDistribution),
            PreserveTypeProd(ExponentialFamilyDistribution{Bernoulli})
        )
            @test prod(strategy, efleft, efright) == ExponentialFamilyDistribution(Bernoulli, ηleft + ηright)
        end

        @test prod(PreserveTypeProd(Bernoulli), efleft, efright) ≈
              prod(ClosedProd(), Bernoulli(pleft), Bernoulli(pright))
    end

    # transformation(logprobability) = exp(logprobability[1]) / (one(Float64) + exp(logprobability[1]))

    # @testset "fisherinformation" begin
    #     for p in 0.1:0.1:0.9
    #         dist = Bernoulli(p)
    #         ef = convert(ExponentialFamilyDistribution, dist)
    #         η = getnaturalparameters(ef)

    #         f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Bernoulli, η))
    #         autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
    #         @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-8
    #         J = ForwardDiff.gradient(transformation, η)
    #         @test J' * fisherinformation(dist) * J ≈ first(fisherinformation(ef)) atol = 1e-8
    #     end
    # end

end

end
