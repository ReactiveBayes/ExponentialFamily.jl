module ExponentialFamilyDistributionTest

using ExponentialFamily, Test, StatsFuns
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, getconditioner
import Distributions: pdf, logpdf, cdf
@testset "ExponentialFamilyDistribution" begin
    params1 = ExponentialFamilyDistribution(Bernoulli, [0.9])
    params2 = ExponentialFamilyDistribution(Bernoulli, [0.2])
    @test getnaturalparameters(params1) == [0.9]
    @test_throws AssertionError ExponentialFamilyDistribution(Bernoulli, [0.9, 0.1])

    @test getnaturalparameters(params1 + params2) == [1.1]
    @test getnaturalparameters(params1 - params2) == [0.7]
    logprobability1 = getindex(getnaturalparameters(params1), 1)
    @test Base.convert(Bernoulli, params1) == Bernoulli(exp(logprobability1) / (1 + exp(logprobability1)))
    @test Base.convert(ExponentialFamilyDistribution, Bernoulli(0.9)) == ExponentialFamilyDistribution(Bernoulli, [logit(0.9)])

    @test_throws AssertionError ExponentialFamilyDistribution(Categorical, log.([0.9, 0.1]), 2.0)
    f = x -> x^3
    @test_throws AssertionError ExponentialFamilyDistribution(Categorical, log.([0.9, 0.1]), f)
end

@testset "pdf,cdf" begin
    params1 = ExponentialFamilyDistribution(Bernoulli, [0.9])

    @test logpdf(params1, 1) == logpdf(Base.convert(Bernoulli, params1), 1)
    @test pdf(params1, 1) == pdf(Base.convert(Bernoulli, params1), 1)
    @test cdf(params1, 1) == cdf(Base.convert(Bernoulli, params1), 1)

    @test logpdf(params1, 0) == logpdf(Base.convert(Bernoulli, params1), 0)
    @test pdf(params1, 0) == pdf(Base.convert(Bernoulli, params1), 0)
    @test cdf(params1, 0) == cdf(Base.convert(Bernoulli, params1), 0)

    @test logpdf(params1, 0.1) == logpdf(Base.convert(Bernoulli, params1), 0.1)
    @test pdf(params1, 0.1) == pdf(Base.convert(Bernoulli, params1), 0.1)
    @test cdf(params1, 0.1) == cdf(Base.convert(Bernoulli, params1), 0.1)
end

end
