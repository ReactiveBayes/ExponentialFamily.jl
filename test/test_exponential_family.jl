module KnownExponentialFamilyDistributionTest

using ExponentialFamily, Test, StatsFuns

import Distributions: RealInterval, ContinuousUnivariateDistribution, Univariate
import ExponentialFamily: basemeasure, sufficientstatistics, logpartition, insupport, ConstantBaseMeasure
import ExponentialFamily: getnaturalparameters, getbasemeasure, getsufficientstatistics, getlogpartition, getsupport
import ExponentialFamily: ExponentialFamilyDistributionAttributes
import ExponentialFamily: paramfloattype, convert_paramfloattype

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

ExponentialFamily.isbasemeasureconstant(::Type{ArbitraryDistributionFromExponentialFamily}) = ConstantBaseMeasure()

ExponentialFamily.getbasemeasure(::Type{ArbitraryDistributionFromExponentialFamily}) = (x) -> oneunit(x)
ExponentialFamily.getsufficientstatistics(::Type{ArbitraryDistributionFromExponentialFamily}) =
    ((x) -> x, (x) -> log(x))
ExponentialFamily.getlogpartition(::Type{ArbitraryDistributionFromExponentialFamily}) = (η) -> 1 / sum(η)
ExponentialFamily.getsupport(::Type{ArbitraryDistributionFromExponentialFamily}) = RealInterval(0, Inf)

ExponentialFamily.vague(::Type{ArbitraryDistributionFromExponentialFamily}) =
    ArbitraryDistributionFromExponentialFamily()

function Base.convert(::Type{ExponentialFamilyDistribution}, ::ArbitraryDistributionFromExponentialFamily)
    return ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [2.0, 2.0])
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
            ExponentialFamilyDistribution(Univariate, [2.0, 2.0], nothing, ArbitraryExponentialFamilyAttributes)
        η = getnaturalparameters(member)

        @test basemeasure(member, 2.0) ≈ 0.5
        @test getbasemeasure(member)(2.0) ≈ 0.5
        @test getbasemeasure(member)(4.0) ≈ 0.25

        @test all(sufficientstatistics(member, 2.0) .≈ (2.0, log(2.0)))
        @test all(map(f -> f(2.0), getsufficientstatistics(member)) .≈ (2.0, log(2.0)))
        @test all(map(f -> f(4.0), getsufficientstatistics(member)) .≈ (4.0, log(4.0)))

        @test logpartition(member) ≈ 0.25
        @test getlogpartition(member)([2.0, 2.0]) ≈ 0.25
        @test getlogpartition(member)([4.0, 4.0]) ≈ 0.125

        @test getsupport(member) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)

        _similar = similar(member)

        @test _similar isa typeof(member)

        # `similar` most probably returns the un-initialized natural parameters with garbage in it
        # But we do expect the functions to work anyway given proper values
        @test basemeasure(_similar, 2.0) ≈ 0.5
        @test all(sufficientstatistics(_similar, 2.0) .≈ (2.0, log(2.0)))
        @test logpartition(_similar, η) ≈ 0.25
        @test getsupport(_similar) == RealInterval(0, Inf)
    end
end

@testset "ExponentialFamilyDistribution" begin

    # See the `ArbitraryDistributionFromExponentialFamily` defined in the fixtures (above)
    @testset let member = ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [2.0, 2.0])
        η = getnaturalparameters(member)

        @test basemeasure(member, 2.0) ≈ 1.0
        @test getbasemeasure(member)(2.0) ≈ 1.0
        @test getbasemeasure(member)(4.0) ≈ 1.0

        @test all(sufficientstatistics(member, 2.0) .≈ (2.0, log(2.0)))
        @test all(map(f -> f(2.0), getsufficientstatistics(member)) .≈ (2.0, log(2.0)))
        @test all(map(f -> f(4.0), getsufficientstatistics(member)) .≈ (4.0, log(4.0)))

        @test logpartition(member) ≈ 0.25
        @test getlogpartition(member)([2.0, 2.0]) ≈ 0.25
        @test getlogpartition(member)([4.0, 4.0]) ≈ 0.125

        @test getsupport(member) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)

        # Computed by hand
        @test logpdf(member, 2.0) ≈ (3.75 + 2log(2))
        @test logpdf(member, 4.0) ≈ (7.75 + 4log(2))
        @test pdf(member, 2.0) ≈ exp(3.75 + 2log(2))
        @test pdf(member, 4.0) ≈ exp(7.75 + 4log(2))

        @test member == member
        @test member ≈ member

        _similar = similar(member)      
        _prod = ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [4.0, 4.0])

        @test prod(ClosedProd(), member, member) == _prod
        @test prod(GenericProd(), member, member) == _prod
        @test prod(PreserveTypeProd(ExponentialFamilyDistribution), member, member) == _prod
        @test prod(PreserveTypeLeftProd(), member, member) == _prod
        @test prod(PreserveTypeRightProd(), member, member) == _prod

        # Test that the generic prod version does not allocate as much as simply creating a similar ef member
        # This is important, because the generic prod version should simply call the in-place version
        @test @allocated(prod(ClosedProd(), member, member)) == @allocated(similar(member))
        @test @allocated(prod(GenericProd(), member, member)) == @allocated(similar(member))
        @test @allocated(prod(PreserveTypeProd(ExponentialFamilyDistribution), member, member)) == @allocated(similar(member))

        @test prod!(_similar, member, member) == _prod
        
        # Test that the in-place prod preserves the container paramfloatype
        for F in (Float16, Float32, Float64)
            @test paramfloattype(prod!(similar(member, F), member, member)) === F
            @test prod!(similar(member, F), member, member) == convert_paramfloattype(F, _prod)
        end

        # Test that the generic in-place prod! version does not allocate at all
        @test @allocated(prod!(_similar, member, member)) === 0
        
    end

    @test vague(ExponentialFamilyDistribution{ArbitraryDistributionFromExponentialFamily}) isa
          ExponentialFamilyDistribution{ArbitraryDistributionFromExponentialFamily}
end

end
