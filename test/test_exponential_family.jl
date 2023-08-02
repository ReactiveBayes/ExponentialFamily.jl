module KnownExponentialFamilyDistributionTest

using ExponentialFamily, Test, StatsFuns

import Distributions: RealInterval, ContinuousUnivariateDistribution, Univariate
import ExponentialFamily: basemeasure, sufficientstatistics, logpartition, insupport
import ExponentialFamily: getnaturalparameters, getbasemeasure, getsufficientstatistics, getlogpartition, getsupport
import ExponentialFamily: ExponentialFamilyDistributionAttributes

# import ExponentialFamily:
#     ExponentialFamilyDistribution, getnaturalparameters, getconditioner, reconstructargument!, as_vec,
#     pack_naturalparameters, unpack_naturalparameters, insupport
# import Distributions: pdf, logpdf, cdf

## ===========================================================================
## Tests fixtures
const ArbitraryExponentialFamilyAttributes = ExponentialFamilyDistributionAttributes((x) -> 1 / sum(x), [ (x) -> x, (x) -> log.(x) ], (x) -> 1 / sum(x),  RealInterval(0, Inf)) 

# Defines its own 
# - `basemeasure`
# - `sufficientstatistics`
# - `logpartition`
# - `support`
struct ArbitraryDistributionFromExponentialFamily <: ContinuousUnivariateDistribution end

ExponentialFamily.check_valid_natural(::Type{ArbitraryDistributionFromExponentialFamily}, η) = true
ExponentialFamily.check_valid_conditioner(::Type{ArbitraryDistributionFromExponentialFamily}, ::Nothing) = true

ExponentialFamily.getbasemeasure(::Type{ArbitraryDistributionFromExponentialFamily}) = (η) -> 1 / sum(η)
ExponentialFamily.getsufficientstatistics(::Type{ArbitraryDistributionFromExponentialFamily}) = [ (η) -> η, (η) -> log.(η) ]
ExponentialFamily.getlogpartition(::Type{ArbitraryDistributionFromExponentialFamily}) = (η) -> 1 / sum(η)
ExponentialFamily.getsupport(::Type{ArbitraryDistributionFromExponentialFamily}) = RealInterval(0, Inf)

## ===========================================================================
## Tests

@testset "ExponentialFamilyDistributionAttributes" begin 

    # See the `ArbitraryExponentialFamilyAttributes` defined in the fixtures (above)
    @testset let attributes = ArbitraryExponentialFamilyAttributes
        @test getbasemeasure(attributes)([ 2.0 ]) ≈ 0.5
        @test getsufficientstatistics(attributes)[1]([ 2.0 ]) ≈ [ 2.0 ]
        @test getsufficientstatistics(attributes)[2]([ 2.0 ]) ≈ [ log(2.0) ]
        @test getlogpartition(attributes)([ 2.0 ]) ≈ 0.5
        @test getsupport(attributes) == RealInterval(0, Inf)
        @test insupport(attributes, 1.0)
        @test !insupport(attributes, -1.0)
    end

    @testset let member = ExponentialFamilyDistribution(Univariate, [ 2.0 ], nothing, ArbitraryExponentialFamilyAttributes)
        η = getnaturalparameters(member)

        @test basemeasure(member) ≈ 0.5
        @test getbasemeasure(member)(η) ≈ 0.5
        @test getbasemeasure(member)([ 4.0 ]) ≈ 0.25

        @test sufficientstatistics(member) ≈ [ [ 2.0 ], [ log(2.0) ] ]
        @test map(f -> f(η), getsufficientstatistics(member)) ≈ [ [ 2.0 ], [ log(2.0) ] ]
        @test map(f -> f([ 4.0 ]), getsufficientstatistics(member)) ≈ [ [ 4.0 ], [ log(4.0) ] ]

        @test logpartition(member) ≈ 0.5
        @test getlogpartition(member)(η) ≈ 0.5
        @test getlogpartition(member)([ 4.0 ]) ≈ 0.25

        @test getsupport(member) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)
    end 

end

@testset "ExponentialFamilyDistribution" begin 

    # See the `ArbitraryDistributionFromExponentialFamily` defined in the fixtures (above)
    @testset let member = ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [ 2.0 ])
        η = getnaturalparameters(member)

        @test basemeasure(member) ≈ 0.5
        @test getbasemeasure(member)(η) ≈ 0.5
        @test getbasemeasure(member)([ 4.0 ]) ≈ 0.25

        @test sufficientstatistics(member) ≈ [ [ 2.0 ], [ log(2.0) ] ]
        @test map(f -> f(η), getsufficientstatistics(member)) ≈ [ [ 2.0 ], [ log(2.0) ] ]
        @test map(f -> f([ 4.0 ]), getsufficientstatistics(member)) ≈ [ [ 4.0 ], [ log(4.0) ] ]

        @test logpartition(member) ≈ 0.5
        @test getlogpartition(member)(η) ≈ 0.5
        @test getlogpartition(member)([ 4.0 ]) ≈ 0.25

        @test getsupport(member) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)
    end

end

# @testset "ExponentialFamilyDistribution" begin
#     ef1 = ExponentialFamilyDistribution(Bernoulli, [0.9])
#     ef2 = ExponentialFamilyDistribution(Bernoulli, [0.2])
#     @test getnaturalparameters(ef1) == [0.9]
#     @test_throws AssertionError ExponentialFamilyDistribution(Bernoulli, [0.9, 0.1])

#     @test getnaturalparameters(ef1) + getnaturalparameters(ef2) == [1.1]
#     @test getnaturalparameters(ef1) - getnaturalparameters(ef2) == [0.7]
#     (logprobability1,) = unpack_naturalparameters(ef1)
#     @test Base.convert(Bernoulli, ef1) == Bernoulli(exp(logprobability1) / (1 + exp(logprobability1)))
#     @test Base.convert(ExponentialFamilyDistribution, Bernoulli(0.9)) ==
#           ExponentialFamilyDistribution(Bernoulli, [logit(0.9)])

#     @test_throws AssertionError ExponentialFamilyDistribution(Categorical, log.([0.9, 0.1]), 2.0)
#     f = x -> x^3
#     @test_throws AssertionError ExponentialFamilyDistribution(Categorical, log.([0.9, 0.1]), f)

#     @test insupport(ef1, 1) == true
#     @test insupport(ef1, 0) == true
# end

# @testset "pdf,cdf" begin
#     ef1 = ExponentialFamilyDistribution(Bernoulli, [0.9])

#     @test logpdf(ef1, 1) ≈ logpdf(Base.convert(Bernoulli, ef1), 1)
#     @test pdf(ef1, 1) ≈ pdf(Base.convert(Bernoulli, ef1), 1)
#     @test cdf(ef1, 1) == cdf(Base.convert(Bernoulli, ef1), 1)

#     @test logpdf(ef1, 0) ≈ logpdf(Base.convert(Bernoulli, ef1), 0)
#     @test pdf(ef1, 0) ≈ pdf(Base.convert(Bernoulli, ef1), 0)
#     @test cdf(ef1, 0) == cdf(Base.convert(Bernoulli, ef1), 0)

#     @test cdf(ef1, 0.1) == cdf(Base.convert(Bernoulli, ef1), 0.1)
# end

end
