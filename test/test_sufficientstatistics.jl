module SufficientStatisticsTest

using Test
using Distributions
import ExponentialFamily: KnownExponentialFamilyDistribution,sufficientstatistics
import StatsFuns: logit

@testset "Sufficient Statistics for KnownExponentialFamilyDistribution" begin
    bernoullief = KnownExponentialFamilyDistribution(Bernoulli, log(0.1))
    @test sufficientstatistics(bernoullief, 1) == 1
    @test sufficientstatistics(bernoullief, 0) == 0
    @test_throws AssertionError sufficientstatistics(bernoullief, 0.1)

    betaef = KnownExponentialFamilyDistribution(Beta, [1, 0.2])
    @test sufficientstatistics(betaef, 0.1) == [log(0.1), log(1.0 - 0.1)]
    @test sufficientstatistics(betaef, 0.9) == [log(0.9), log(1.0 - 0.9)]
    @test sufficientstatistics(betaef, 0.999) == [log(0.999), log(1.0 - 0.999)]
    @test_throws AssertionError sufficientstatistics(betaef, 1.01)
    @test_throws AssertionError sufficientstatistics(betaef, -0.01)

    binomialef = KnownExponentialFamilyDistribution(Binomial, logit(0.3), 10)
    @test sufficientstatistics(binomialef, 1) == 1
    @test sufficientstatistics(binomialef, 7) == 7
    @test_throws AssertionError sufficientstatistics(binomialef, 11)
    @test_throws AssertionError sufficientstatistics(binomialef, 1.1)
end



end