module BernoulliTest

using Test
using ExponentialFamily
using Distributions
using Random
using StatsFuns
import ExponentialFamily:
    KnownExponentialFamilyDistribution, getnaturalparameters, compute_logscale, logpartition, basemeasure

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
        @test logpartition(convert(KnownExponentialFamilyDistribution, Bernoulli(0.5))) ≈ log(2)
        b_99 = Bernoulli(0.99)
        for i in 1:9
            b = Bernoulli(i / 10.0)
            bnp = convert(KnownExponentialFamilyDistribution, b)
            @test convert(Distribution, bnp) ≈ b
            @test logpdf(bnp, 1) ≈ logpdf(b, 1)
            @test logpdf(bnp, 0) ≈ logpdf(b, 0)

            @test convert(KnownExponentialFamilyDistribution, b) ==
                  KnownExponentialFamilyDistribution(Bernoulli, [logit(i / 10.0)])
        end
        @test isproper(KnownExponentialFamilyDistribution(Bernoulli, [10])) === true
        @test basemeasure(b_99, 0.1) == 1.0
        @test basemeasure(KnownExponentialFamilyDistribution(Bernoulli, [10]), 0.2) == 1.0
    end

    @testset "prod with KnownExponentialFamilyDistribution" begin
        for pleft=0.01:0.01:0.99
            ηleft  = log(pleft/(1-pleft))
            efleft = KnownExponentialFamilyDistribution(Bernoulli, ηleft)
            for pright = 0.01:0.01:0.99
                ηright = log(pright/(1-pright))
                efright = KnownExponentialFamilyDistribution(Bernoulli, ηright)
                @test prod(ClosedProd(), efleft,efright) == KnownExponentialFamilyDistribution(Bernoulli, ηleft+ηright)
                @test prod(efleft,efright) == KnownExponentialFamilyDistribution(Bernoulli, ηleft+ηright)
            end
        end
    end

    @testset "prod with Distribution" begin
        @test prod(ClosedProd(), Bernoulli(0.5), Bernoulli(0.5)) ≈ Bernoulli(0.5)
        @test prod(ClosedProd(), Bernoulli(0.1), Bernoulli(0.6)) ≈ Bernoulli(0.14285714285714285)
        @test prod(ClosedProd(), Bernoulli(0.78), Bernoulli(0.05)) ≈ Bernoulli(0.1572580645161291)
    end
end

end
