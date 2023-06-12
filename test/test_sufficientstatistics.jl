module SufficientStatisticsTest

using Test
using Distributions
import ExponentialFamily: KnownExponentialFamilyDistribution,sufficientstatistics

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

    
end



end