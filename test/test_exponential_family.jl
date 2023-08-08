module KnownExponentialFamilyDistributionTest

using ExponentialFamily, Distributions, Test, StatsFuns, BenchmarkTools

import Distributions: RealInterval, ContinuousUnivariateDistribution, Univariate
import ExponentialFamily: basemeasure, sufficientstatistics, logpartition, insupport, ConstantBaseMeasure
import ExponentialFamily: getnaturalparameters, getbasemeasure, getsufficientstatistics, getlogpartition, getsupport
import ExponentialFamily: ExponentialFamilyDistributionAttributes, NaturalParametersSpace
import ExponentialFamily: paramfloattype, convert_paramfloattype

# import ExponentialFamily:
#     ExponentialFamilyDistribution, getnaturalparameters, getconditioner, reconstructargument!, as_vec,
#     pack_naturalparameters, unpack_naturalparameters, insupport
# import Distributions: pdf, logpdf, cdf

## ===========================================================================
## Tests fixtures
const ArbitraryExponentialFamilyAttributes = ExponentialFamilyDistributionAttributes(
    (x) -> 1 / x,
    ((x) -> x, (x) -> log(x)),
    (η) -> 1 / sum(η),
    RealInterval(0, Inf)
)

# Defines its own 
# - `basemeasure`
# - `sufficientstatistics`
# - `logpartition`
# - `support`
struct ArbitraryDistributionFromExponentialFamily <: ContinuousUnivariateDistribution 
    p1 :: Float64
    p2 :: Float64
end

ExponentialFamily.isproper(::NaturalParametersSpace, ::Type{ArbitraryDistributionFromExponentialFamily}, η, conditioner) = isnothing(conditioner)
ExponentialFamily.isbasemeasureconstant(::Type{ArbitraryDistributionFromExponentialFamily}) = ConstantBaseMeasure()
ExponentialFamily.getbasemeasure(::Type{ArbitraryDistributionFromExponentialFamily}) = (x) -> oneunit(x)
ExponentialFamily.getsufficientstatistics(::Type{ArbitraryDistributionFromExponentialFamily}) =
    ((x) -> x, (x) -> log(x))
ExponentialFamily.getlogpartition(::NaturalParametersSpace, ::Type{ArbitraryDistributionFromExponentialFamily}) = (η) -> 1 / sum(η)
ExponentialFamily.getsupport(::Type{ArbitraryDistributionFromExponentialFamily}) = RealInterval(0, Inf)

ExponentialFamily.vague(::Type{ArbitraryDistributionFromExponentialFamily}) =
    ArbitraryDistributionFromExponentialFamily(1.0, 1.0)

Distributions.params(dist::ArbitraryDistributionFromExponentialFamily) = (dist.p1, dist.p2)

(::MeanToNatural{ArbitraryDistributionFromExponentialFamily})(params::Tuple) = (params[1] + 1, params[2] + 1)
(::NaturalToMean{ArbitraryDistributionFromExponentialFamily})(params::Tuple) = (params[1] - 1, params[2] - 1)

ExponentialFamily.unpack_parameters(::Type{ArbitraryDistributionFromExponentialFamily}, η) = (η[1], η[2], )

# Conditional member of exponential family
struct ArbitraryConditionedDistributionFromExponentialFamily <: ContinuousUnivariateDistribution 
    con :: Int
    p1 :: Float64
end

ExponentialFamily.isproper(::NaturalParametersSpace, ::Type{ArbitraryConditionedDistributionFromExponentialFamily}, η, conditioner) = isinteger(conditioner)
ExponentialFamily.isbasemeasureconstant(::Type{ArbitraryConditionedDistributionFromExponentialFamily}) = NonConstantBaseMeasure()
ExponentialFamily.getbasemeasure(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, conditioner) = (x) -> x ^ conditioner
ExponentialFamily.getsufficientstatistics(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, conditioner) =
    ((x) -> log(x - conditioner), )
ExponentialFamily.getlogpartition(::NaturalParametersSpace, ::Type{ArbitraryConditionedDistributionFromExponentialFamily}, conditioner) = (η) -> conditioner / sum(η)
ExponentialFamily.getsupport(::Type{ArbitraryConditionedDistributionFromExponentialFamily}) = RealInterval(0, Inf)

ExponentialFamily.vague(::Type{ArbitraryConditionedDistributionFromExponentialFamily}) =
ArbitraryConditionedDistributionFromExponentialFamily(1.0, -2)

Distributions.params(dist::ArbitraryConditionedDistributionFromExponentialFamily) = (dist.con, dist.p1)

ExponentialFamily.separate_conditioner(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, params) = ((params[2], ), params[1])
ExponentialFamily.join_conditioner(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, cparams, conditioner) = (conditioner, cparams...)

(::MeanToNatural{ArbitraryConditionedDistributionFromExponentialFamily})(params::Tuple, conditioner::Number) = (params[1] + conditioner,)
(::NaturalToMean{ArbitraryConditionedDistributionFromExponentialFamily})(params::Tuple, conditioner::Number) = (params[1] - conditioner,)

ExponentialFamily.unpack_parameters(::Type{ArbitraryConditionedDistributionFromExponentialFamily}, η) = (η[1], )



## ===========================================================================
## Tests

@testset "ExponentialFamilyDistributionAttributes" begin

    # See the `ArbitraryExponentialFamilyAttributes` defined in the fixtures (above)
    @testset let attributes = ArbitraryExponentialFamilyAttributes
        @test @inferred(getbasemeasure(attributes)(2.0)) ≈ 0.5
        @test @inferred(getsufficientstatistics(attributes)[1](2.0)) ≈ 2.0
        @test @inferred(getsufficientstatistics(attributes)[2](2.0)) ≈ log(2.0)
        @test @inferred(getlogpartition(attributes)([2.0])) ≈ 0.5
        @test @inferred(getsupport(attributes)) == RealInterval(0, Inf)
        @test @inferred(insupport(attributes, 1.0))
        @test !@inferred(insupport(attributes, -1.0))
    end

    @testset let member =
            ExponentialFamilyDistribution(Univariate, [2.0, 2.0], nothing, ArbitraryExponentialFamilyAttributes)
        η = @inferred(getnaturalparameters(member))

        @test @inferred(basemeasure(member, 2.0)) ≈ 0.5
        @test @inferred(getbasemeasure(member)(2.0)) ≈ 0.5
        @test @inferred(getbasemeasure(member)(4.0)) ≈ 0.25

        @test all(@inferred(sufficientstatistics(member, 2.0)) .≈ (2.0, log(2.0)))
        @test all(@inferred(map(f -> f(2.0), getsufficientstatistics(member))) .≈ (2.0, log(2.0)))
        @test all(@inferred(map(f -> f(4.0), getsufficientstatistics(member))) .≈ (4.0, log(4.0)))

        @test @inferred(logpartition(member)) ≈ 0.25
        @test @inferred(getlogpartition(member)([2.0, 2.0])) ≈ 0.25
        @test @inferred(getlogpartition(member)([4.0, 4.0])) ≈ 0.125

        @test @inferred(getsupport(member)) == RealInterval(0, Inf)
        @test @inferred(insupport(member, 1.0))
        @test !@inferred(insupport(member, -1.0))

        _similar = @inferred(similar(member))

        # The standard `@allocated` is not really reliable in this test 
        # We avoid using the `BenchmarkTools`, but here it is essential
        @test @ballocated(logpdf($member, 1.0)) === 0
        @test @ballocated(pdf($member, 1.0)) === 0

        @test _similar isa typeof(member)

        # `similar` most probably returns the un-initialized natural parameters with garbage in it
        # But we do expect the functions to work anyway given proper values
        @test @inferred(basemeasure(_similar, 2.0)) ≈ 0.5
        @test all(@inferred(sufficientstatistics(_similar, 2.0)) .≈ (2.0, log(2.0)))
        @test @inferred(logpartition(_similar, η)) ≈ 0.25
        @test @inferred(getsupport(_similar)) == RealInterval(0, Inf)
    end
end

@testset "ExponentialFamilyDistribution" begin

    # See the `ArbitraryDistributionFromExponentialFamily` defined in the fixtures (above)
    @testset for member in (ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [2.0, 2.0]), convert(ExponentialFamilyDistribution, ArbitraryDistributionFromExponentialFamily(1.0, 1.0)))
        η = @inferred(getnaturalparameters(member))

        @test convert(ExponentialFamilyDistribution, convert(Distribution, member)) == ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [2.0, 2.0])
        @test convert(Distribution, convert(ExponentialFamilyDistribution, member)) == ArbitraryDistributionFromExponentialFamily(1.0, 1.0)

        @test @inferred(basemeasure(member, 2.0)) ≈ 1.0
        @test @inferred(getbasemeasure(member)(2.0)) ≈ 1.0
        @test @inferred(getbasemeasure(member)(4.0)) ≈ 1.0

        @test all(@inferred(sufficientstatistics(member, 2.0)) .≈ (2.0, log(2.0)))
        @test all(@inferred(map(f -> f(2.0), getsufficientstatistics(member))) .≈ (2.0, log(2.0)))
        @test all(@inferred(map(f -> f(4.0), getsufficientstatistics(member))) .≈ (4.0, log(4.0)))

        @test @inferred(logpartition(member)) ≈ 0.25
        @test @inferred(getlogpartition(member)([2.0, 2.0])) ≈ 0.25
        @test @inferred(getlogpartition(member)([4.0, 4.0])) ≈ 0.125

        @test @inferred(getsupport(member)) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)

        # Computed by hand
        @test @inferred(logpdf(member, 2.0)) ≈ (3.75 + 2log(2))
        @test @inferred(logpdf(member, 4.0)) ≈ (7.75 + 4log(2))
        @test @inferred(pdf(member, 2.0)) ≈ exp(3.75 + 2log(2))
        @test @inferred(pdf(member, 4.0)) ≈ exp(7.75 + 4log(2))

        # The standard `@allocated` is not really reliable in this test 
        # We avoid using the `BenchmarkTools`, but here it is essential
        @test @ballocated(logpdf($member, 2.0)) === 0
        @test @ballocated(pdf($member, 2.0)) === 0

        @test @inferred(member == member)
        @test @inferred(member ≈ member)

        _similar = @inferred(similar(member))
        _prod = ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [4.0, 4.0])

        @test @inferred(prod(ClosedProd(), member, member)) == _prod
        @test @inferred(prod(GenericProd(), member, member)) == _prod
        @test @inferred(prod(PreserveTypeProd(ExponentialFamilyDistribution), member, member)) == _prod
        @test @inferred(prod(PreserveTypeLeftProd(), member, member)) == _prod
        @test @inferred(prod(PreserveTypeRightProd(), member, member)) == _prod

        # Test that the generic prod version does not allocate as much as simply creating a similar ef member
        # This is important, because the generic prod version should simply call the in-place version
        @test @allocated(prod(ClosedProd(), member, member)) <= @allocated(similar(member))
        @test @allocated(prod(GenericProd(), member, member)) <= @allocated(similar(member))
        # @test @allocated(prod(PreserveTypeProd(ExponentialFamilyDistribution), member, member)) <=
        #       @allocated(similar(member))

        @test @inferred(prod!(_similar, member, member)) == _prod

        # Test that the in-place prod preserves the container paramfloatype
        for F in (Float16, Float32, Float64)
            @test @inferred(paramfloattype(prod!(similar(member, F), member, member))) === F
            @test @inferred(prod!(similar(member, F), member, member)) == convert_paramfloattype(F, _prod)
        end

        # Test that the generic in-place prod! version does not allocate at all
        @test @allocated(prod!(_similar, member, member)) === 0
    end

    @test @inferred(vague(ExponentialFamilyDistribution{ArbitraryDistributionFromExponentialFamily})) isa
          ExponentialFamilyDistribution{ArbitraryDistributionFromExponentialFamily}

    # See the `ArbitraryDistributionFromExponentialFamily` defined in the fixtures (above)
    # p1 = 3.0, con = -2
    @testset for member in (ExponentialFamilyDistribution(ArbitraryConditionedDistributionFromExponentialFamily, [1.0], -2), convert(ExponentialFamilyDistribution, ArbitraryConditionedDistributionFromExponentialFamily(-2, 3.0)))
        η = @inferred(getnaturalparameters(member))

        @test convert(ExponentialFamilyDistribution, convert(Distribution, member)) == ExponentialFamilyDistribution(ArbitraryConditionedDistributionFromExponentialFamily, [1.0, ], -2)
        @test convert(Distribution, convert(ExponentialFamilyDistribution, member)) == ArbitraryConditionedDistributionFromExponentialFamily(-2, 3.0)

        @test @inferred(basemeasure(member, 2.0)) ≈ 2.0 ^ -2
        @test @inferred(getbasemeasure(member)(2.0)) ≈ 2.0 ^ -2
        @test @inferred(getbasemeasure(member)(4.0)) ≈ 4.0 ^ -2

        @test all(@inferred(sufficientstatistics(member, 2.0)) .≈ (log(2.0 + 2), ))
        @test all(@inferred(map(f -> f(2.0), getsufficientstatistics(member))) .≈ (log(2.0 + 2), ))
        @test all(@inferred(map(f -> f(4.0), getsufficientstatistics(member))) .≈ (log(4.0 + 2), ))

        @test @inferred(logpartition(member)) ≈ -2.0
        @test @inferred(getlogpartition(member)([2.0, ])) ≈ -1.0
        @test @inferred(getlogpartition(member)([4.0, ])) ≈ -0.5

        @test @inferred(getsupport(member)) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)

        # # Computed by hand
        @test @inferred(logpdf(member, 2.0)) ≈ (log(2.0 ^ -2) + log(2.0 + 2) + 2.0)
        @test @inferred(logpdf(member, 4.0)) ≈ (log(4.0 ^ -2) + log(4.0 + 2) + 2.0)
        @test @inferred(pdf(member, 2.0)) ≈ exp((log(2.0 ^ -2) + log(2.0 + 2) + 2.0))
        @test @inferred(pdf(member, 4.0)) ≈ exp((log(4.0 ^ -2) + log(4.0 + 2) + 2.0))

        # The standard `@allocated` is not really reliable in this test 
        # We avoid using the `BenchmarkTools`, but here it is essential
        @test @ballocated(logpdf($member, 2.0)) === 0
        @test @ballocated(pdf($member, 2.0)) === 0

        @test @inferred(member == member)
        @test @inferred(member ≈ member)

        _similar = @inferred(similar(member))
        _prod = ExponentialFamilyDistribution(ArbitraryConditionedDistributionFromExponentialFamily, [1.0, ], -2)

        # We don't test the prod becasue the basemeasure is not a constant, so the generic prod is not applicable

        # # Test that the in-place prod preserves the container paramfloatype
        for F in (Float16, Float32, Float64)
            @test @inferred(paramfloattype(similar(member, F))) === F
        end
    end

    @test @inferred(vague(ExponentialFamilyDistribution{ArbitraryConditionedDistributionFromExponentialFamily})) isa
          ExponentialFamilyDistribution{ArbitraryConditionedDistributionFromExponentialFamily}
end

end
