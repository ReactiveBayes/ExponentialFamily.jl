module BernoulliTest

using Test
using ExponentialFamily
using Distributions
using Random
using StatsFuns
import ExponentialFamily: NaturalParameters,get_params,compute_logscale,lognormalizer

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

    @testset "prod Bernoulli-Bernoulli" begin
        @test prod(ProdAnalytical(), Bernoulli(0.5), Bernoulli(0.5)) ≈ Bernoulli(0.5)
        @test prod(ProdAnalytical(), Bernoulli(0.1), Bernoulli(0.6)) ≈ Bernoulli(0.14285714285714285)
        @test prod(ProdAnalytical(), Bernoulli(0.78), Bernoulli(0.05)) ≈ Bernoulli(0.1572580645161291)
    end

    @testset "probvec" begin
        @test probvec(Bernoulli(0.5)) === (0.5, 0.5)
        @test probvec(Bernoulli(0.3)) === (0.7, 0.3)
        @test probvec(Bernoulli(0.6)) === (0.4, 0.6)
    end

    @testset "prod logscale Bernoulli-Bernoulli/Categorical" begin
        @test compute_logscale(Bernoulli(0.5), Bernoulli(0.5), Bernoulli(0.5)) ≈ log(0.5)
        @test compute_logscale(Bernoulli(1), Bernoulli(0.5), Bernoulli(1)) ≈ log(0.5)
        @test compute_logscale(Categorical([0.5, 0.5]), Bernoulli(0.5), Categorical([0.5, 0.5])) ≈ log(0.5)
        @test compute_logscale(Categorical([0.5, 0.5]), Categorical([0.5, 0.5]), Bernoulli(0.5)) ≈ log(0.5)
        @test compute_logscale(Categorical([1.0, 0.0]), Bernoulli(0.5), Categorical([1])) ≈ log(0.5)
        @test compute_logscale(Categorical([1.0, 0.0, 0.0]), Bernoulli(0.5), Categorical([1.0, 0, 0])) ≈ log(0.5)
    end

    @testset "NaturalParameters" begin
        @test lognormalizer(convert(NaturalParameters,Bernoulli(0.5))) ≈ log(2)
        b_99 = Bernoulli(0.99)
        for i in 1:9
            b = Bernoulli(i / 10.0)
            bnp = convert(NaturalParameters,b)
            @test convert(Distribution, bnp) ≈ b
            @test logpdf(bnp, 1) ≈ logpdf(b, 1)
            @test logpdf(bnp, 0) ≈ logpdf(b, 0)

            @test convert(NaturalParameters, b) == NaturalParameters(Bernoulli,[logit(i / 10.0)])

            @test prod(ProdAnalytical(), convert(Distribution, convert(NaturalParameters,b_99) - bnp), b) ≈ b_99
        end
        @test isproper(NaturalParameters(Bernoulli,[10])) === true
    end
end

end
