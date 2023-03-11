module NaturalParametersTest

using ExponentialFamily, Test, StatsFuns
import ExponentialFamily: NaturalParameters, get_params
@testset "NaturalParameters" begin
    params1 = NaturalParameters(Bernoulli,[0.9])
    params2 = NaturalParameters(Bernoulli,[0.2])
    @test get_params(params1) == [0.9]
    @test_throws AssertionError NaturalParameters(Bernoulli,[0.9,0.1])


    @test get_params(params1 + params2) == [1.1]
    @test get_params(params1 - params2) == [0.7]
    logprobability1 = getindex(get_params(params1),1)
    @test Base.convert(Bernoulli, params1) == Bernoulli(exp(logprobability1) / (1 + exp(logprobability1)))
    @test Base.convert(NaturalParameters,Bernoulli(0.9)) == NaturalParameters(Bernoulli,[logit(0.9)])
end

@testset "pdf,cdf" begin
    params1 = NaturalParameters(Bernoulli,[0.9])

    @test logpdf(params1, 1) == logpdf(Base.convert(Bernoulli, params1),1)
    @test pdf(params1, 1) == pdf(Base.convert(Bernoulli, params1),1)
    @test cdf(params1, 1) == cdf(Base.convert(Bernoulli, params1),1)

    @test logpdf(params1, 0) == logpdf(Base.convert(Bernoulli, params1),0)
    @test pdf(params1, 0) == pdf(Base.convert(Bernoulli, params1),0)
    @test cdf(params1, 0) == cdf(Base.convert(Bernoulli, params1),0)

    @test logpdf(params1, 0.1) == logpdf(Base.convert(Bernoulli, params1),0.1)
    @test pdf(params1, 0.1) == pdf(Base.convert(Bernoulli, params1),0.1)
    @test cdf(params1, 0.1) == cdf(Base.convert(Bernoulli, params1),0.1)
end


end