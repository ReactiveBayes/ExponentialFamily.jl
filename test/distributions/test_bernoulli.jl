module BernoulliTest

using Test
using ExponentialFamily
using Distributions
using ForwardDiff
using Random
using StatsFuns
import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, compute_logscale, logpartition, basemeasure,
    sufficientstatistics, fisherinformation, pack_naturalparameters,unpack_naturalparameters

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

    @testset "natural parameters related" begin
        @test logpartition(convert(ExponentialFamilyDistribution, Bernoulli(0.5))) ≈ log(2)
        b_99 = Bernoulli(0.99)
        for i in 1:9
            b = Bernoulli(i / 10.0)
            bnp = convert(ExponentialFamilyDistribution, b)
            @test convert(Distribution, bnp) ≈ b
            @test logpdf(bnp, 1) ≈ logpdf(b, 1)
            @test logpdf(bnp, 0) ≈ logpdf(b, 0)

            @test convert(ExponentialFamilyDistribution, b) ==
                  ExponentialFamilyDistribution(Bernoulli, [logit(i / 10.0)])
        end
        @test isproper(ExponentialFamilyDistribution(Bernoulli, [10])) === true
        bernoullief = ExponentialFamilyDistribution(Bernoulli, [log(0.1)])
        @test sufficientstatistics(bernoullief, 1) == 1
        @test sufficientstatistics(bernoullief, 0) == 0
    end

    @testset "prod with ExponentialFamilyDistribution" begin
        for pleft in 0.01:0.01:0.99
            ηleft  = [log(pleft / (1 - pleft))]
            efleft = ExponentialFamilyDistribution(Bernoulli, ηleft)
            for pright in 0.01:0.01:0.99
                ηright = [log(pright / (1 - pright))]
                efright = ExponentialFamilyDistribution(Bernoulli, ηright)
                @test prod(ClosedProd(), efleft, efright) ==
                      ExponentialFamilyDistribution(Bernoulli, ηleft + ηright)
                @test prod(efleft, efright) == ExponentialFamilyDistribution(Bernoulli, ηleft + ηright)
            end
        end
    end

    @testset "prod with Distribution" begin
        @test prod(ClosedProd(), Bernoulli(0.5), Bernoulli(0.5)) ≈ Bernoulli(0.5)
        @test prod(ClosedProd(), Bernoulli(0.1), Bernoulli(0.6)) ≈ Bernoulli(0.14285714285714285)
        @test prod(ClosedProd(), Bernoulli(0.78), Bernoulli(0.05)) ≈ Bernoulli(0.1572580645161291)
    end

    transformation(logprobability) = exp(logprobability[1]) / (one(Float64) + exp(logprobability[1]))
    @testset "fisherinformation" begin
        for p in 0.1:0.1:0.9
            dist = Bernoulli(p)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Bernoulli, η))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test fisherinformation(ef) ≈ first(autograd_information(η)) atol = 1e-8
            J = ForwardDiff.gradient(transformation, η)
            @test first(J' * fisherinformation(dist) * J) ≈ first(fisherinformation(ef)) atol = 1e-8
        end
    end

    @testset "ExponentialFamilyDistribution mean var" begin
        for p in 0.1:0.1:0.9
            dist = Bernoulli(p)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end

end
