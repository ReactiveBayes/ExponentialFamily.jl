# VonMises comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "VonMises: vague" begin
    include("distributions_setuptests.jl")

    d = vague(VonMises)

    @test typeof(d) <: VonMises
    @test mean(d) === 0.0
    @test params(d) === (0.0, 1.0e-12)
end

@testitem "VonMises: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for a in -2:1.0:2, b in 0.1:4.0:10.0
        @testset let d = VonMises(a, b)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false)

            for x in (a-1):0.5:(a+1)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === inv(twoπ)
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (cos(x), sin(x)))
                @test @inferred(logpartition(ef)) ≈ (log(besseli(0, b)))
            end

            @test !@inferred(insupport(ef, -6))
            @test @inferred(insupport(ef, 0.5))

            # Not in the support
            @test_throws Exception logpdf(ef, -6.0)
        end
    end

    # Test failing isproper cases
    @test !isproper(DefaultParametersSpace(), VonMises, [-1])
    @test !isproper(DefaultParametersSpace(), VonMises, [1], 3.0)
    @test !isproper(DefaultParametersSpace(), VonMises, [1, -2])
end

@testitem "VonMises: prod with Distributions" begin
    include("distributions_setuptests.jl")
    for dist1 in [VonMises(10randn(), 10rand()) for _ in 1:20], dist2 in [VonMises(10randn(), 10rand()) for _ in 1:20]
        ef1 = convert(ExponentialFamilyDistribution, dist1)
        ef2 = convert(ExponentialFamilyDistribution, dist2)
        prod_ef = prod(GenericProd(), ef1, ef2)
        for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
            @test prod(strategy, dist1, dist2) ≈ convert(Distribution, prod_ef)
        end
    end
end

@testitem "VonMises: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for kleft in (0.01, 1.0), kright in (0.01, 1.0), alphaleft in 0.1:0.1:0.9, alpharight in 0.1:0.1:0.9
        let left = VonMises(alphaleft, kleft), right = VonMises(alpharight, kright)
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (PreserveTypeProd(ExponentialFamilyDistribution{VonMises}), GenericProd())
            )
        end
    end
end
