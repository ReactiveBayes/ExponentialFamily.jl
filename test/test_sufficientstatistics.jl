module SufficientStatisticsTest

using Test
using Distributions
import ExponentialFamily: KnownExponentialFamilyDistribution,sufficientstatistics

@testset "Sufficient Statistics for KnownExponentialFamilyDistribution" begin
    bernoullief = KnownExponentialFamilyDistribution(Bernoulli, log(0.1))
    @test sufficientstatistics(bernoullief, 1) == 1
    @test sufficientstatistics(bernoullief, 0) == 0
    @test_throws AssertionError sufficientstatistics(bernoullief, 0.1)

end



end