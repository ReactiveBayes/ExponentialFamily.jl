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
const ArbitraryExponentialFamilyAttributes = ExponentialFamilyDistributionAttributes(
    (x) -> 1 / x,
    ((x) -> x, (x) -> log.(x)),
    (η) -> 1 / sum(η),
    RealInterval(0, Inf)
)

# Defines its own 
# - `basemeasure`
# - `sufficientstatistics`
# - `logpartition`
# - `support`
struct ArbitraryDistributionFromExponentialFamily <: ContinuousUnivariateDistribution end

ExponentialFamily.check_valid_natural(::Type{ArbitraryDistributionFromExponentialFamily}, η) = true
ExponentialFamily.check_valid_conditioner(::Type{ArbitraryDistributionFromExponentialFamily}, ::Nothing) = true

ExponentialFamily.getbasemeasure(::Type{ArbitraryDistributionFromExponentialFamily}) = (x) -> 1 / x
ExponentialFamily.getsufficientstatistics(::Type{ArbitraryDistributionFromExponentialFamily}) =
    ((x) -> x, (η) -> log.(η))
ExponentialFamily.getlogpartition(::Type{ArbitraryDistributionFromExponentialFamily}) = (η) -> 1 / sum(η)
ExponentialFamily.getsupport(::Type{ArbitraryDistributionFromExponentialFamily}) = RealInterval(0, Inf)

ExponentialFamily.vague(::Type{ArbitraryDistributionFromExponentialFamily}) =
    ArbitraryDistributionFromExponentialFamily()

function Base.convert(::Type{ExponentialFamilyDistribution}, ::ArbitraryDistributionFromExponentialFamily)
    return ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [2.0])
end

## ===========================================================================
## Tests

@testset "ExponentialFamilyDistributionAttributes" begin

    # See the `ArbitraryExponentialFamilyAttributes` defined in the fixtures (above)
    @testset let attributes = ArbitraryExponentialFamilyAttributes
        @test getbasemeasure(attributes)(2.0) ≈ 0.5
        @test getsufficientstatistics(attributes)[1](2.0) ≈ 2.0
        @test getsufficientstatistics(attributes)[2](2.0) ≈ log(2.0)
        @test getlogpartition(attributes)([2.0]) ≈ 0.5
        @test getsupport(attributes) == RealInterval(0, Inf)
        @test insupport(attributes, 1.0)
        @test !insupport(attributes, -1.0)
    end

    @testset let member =
            ExponentialFamilyDistribution(Univariate, [2.0], nothing, ArbitraryExponentialFamilyAttributes)
        η = getnaturalparameters(member)

        @test basemeasure(member, 2.0) ≈ 0.5
        @test getbasemeasure(member)(2.0) ≈ 0.5
        @test getbasemeasure(member)(4.0) ≈ 0.25

        @test all(sufficientstatistics(member, 2.0) .≈ (2.0, log(2.0)))
        @test all(map(f -> f(2.0), getsufficientstatistics(member)) .≈ (2.0, log(2.0)))
        @test all(map(f -> f(4.0), getsufficientstatistics(member)) .≈ (4.0, log(4.0)))

        @test logpartition(member) ≈ 0.5
        @test getlogpartition(member)([2.0]) ≈ 0.5
        @test getlogpartition(member)([4.0]) ≈ 0.25

        @test getsupport(member) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)

        _similar = similar(member)

        @test _similar isa typeof(member)

        # `similar` most probably returns the un-initialized natural parameters with garbage in it
        # But we do expect the functions to work anyway given proper values
        @test basemeasure(_similar, 2.0) ≈ 0.5
        @test all(sufficientstatistics(_similar, 2.0) .≈ (2.0, log(2.0)))
        @test logpartition(_similar, η) ≈ 0.5
        @test getsupport(_similar) == RealInterval(0, Inf)
    end
end

@testset "ExponentialFamilyDistribution" begin

    # See the `ArbitraryDistributionFromExponentialFamily` defined in the fixtures (above)
    @testset let member = ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [2.0])
        η = getnaturalparameters(member)

        @test basemeasure(member, 2.0) ≈ 0.5
        @test getbasemeasure(member)(2.0) ≈ 0.5
        @test getbasemeasure(member)(4.0) ≈ 0.25

        @test all(sufficientstatistics(member, 2.0) .≈ (2.0, log(2.0)))
        @test all(map(f -> f(2.0), getsufficientstatistics(member)) .≈ (2.0, log(2.0)))
        @test all(map(f -> f(4.0), getsufficientstatistics(member)) .≈ (4.0, log(4.0)))

        @test logpartition(member) ≈ 0.5
        @test getlogpartition(member)([2.0]) ≈ 0.5
        @test getlogpartition(member)([4.0]) ≈ 0.25

        @test getsupport(member) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)

        @test member == member
        @test member ≈ member
    end

    @test vague(ExponentialFamilyDistribution{ArbitraryDistributionFromExponentialFamily}) isa
          ExponentialFamilyDistribution{ArbitraryDistributionFromExponentialFamily}
end

end
