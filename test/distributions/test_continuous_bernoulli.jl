module ContinuousBernoulliTest

using Test
using ExponentialFamily
using Distributions
using Random
using StatsFuns
import ExponentialFamily: NaturalParameters, get_params, compute_logscale, lognormalizer, basemeasure

@testset "ContinuousBernoulli" begin

    # ContinuousBernoulli comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(ContinuousBernoulli)

        @test typeof(d) <: ContinuousBernoulli
        @test mean(d) === 0.5
        @test succprob(d) === 0.5
        @test failprob(d) === 0.5
    end

    @testset "prod ContinuousBernoulli-ContinuousBernoulli" begin
        @test prod(ProdAnalytical(), ContinuousBernoulli(0.5), ContinuousBernoulli(0.5)) ≈ ContinuousBernoulli(0.5)
        @test prod(ProdAnalytical(), ContinuousBernoulli(0.1), ContinuousBernoulli(0.6)) ≈
              ContinuousBernoulli(0.14285714285714285)
        @test prod(ProdAnalytical(), ContinuousBernoulli(0.78), ContinuousBernoulli(0.05)) ≈
              ContinuousBernoulli(0.1572580645161291)
    end

    @testset "probvec" begin
        @test probvec(ContinuousBernoulli(0.5)) === (0.5, 0.5)
        @test probvec(ContinuousBernoulli(0.3)) === (0.7, 0.3)
        @test probvec(ContinuousBernoulli(0.6)) === (0.4, 0.6)
    end

    @testset "NaturalParameters" begin
        @test lognormalizer(convert(NaturalParameters, ContinuousBernoulli(0.5))) ≈ log(2)
        @test lognormalizer(convert(NaturalParameters, ContinuousBernoulli(0.2))) ≈ log((-3 / 4) / log(1 / 4))
        b_99 = ContinuousBernoulli(0.99)
        for i in 1:9
            b = ContinuousBernoulli(i / 10.0)
            bnp = convert(NaturalParameters, b)
            @test convert(Distribution, bnp) ≈ b
            @test logpdf(bnp, 1) ≈ logpdf(b, 1)
            @test logpdf(bnp, 0) ≈ logpdf(b, 0)

            @test convert(NaturalParameters, b) == NaturalParameters(ContinuousBernoulli, [logit(i / 10.0)])

            @test prod(ProdAnalytical(), convert(Distribution, convert(NaturalParameters, b_99) - bnp), b) ≈ b_99
        end
        @test isproper(NaturalParameters(ContinuousBernoulli, [10])) === true
        @test basemeasure(b_99, 0.1) == 1.0
        @test basemeasure(NaturalParameters(ContinuousBernoulli, [10]), 0.2) == 1.0

        @testset "+(::NaturalParameters{ContinuousBernoulli}, ::NaturalParameters{ContinuousBernoulli})" begin
            left = convert(NaturalParameters, ContinuousBernoulli(0.5))
            right = convert(NaturalParameters, ContinuousBernoulli(0.6))
            @test (left + right) == convert(NaturalParameters, ContinuousBernoulli(0.6))
        end
    end
end

end
