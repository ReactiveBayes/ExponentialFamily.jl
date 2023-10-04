
# Beta comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Beta: vague" begin
    include("distributions_setuptests.jl")

    d = vague(Beta)

    @test typeof(d) <: Beta
    @test mean(d) === 0.5
    @test params(d) === (1.0, 1.0)
end

@testitem "Beta: mean(::typeof(log))" begin
    include("distributions_setuptests.jl")

    @test mean(log, Beta(1.0, 3.0)) ≈ -1.8333333333333335
    @test mean(log, Beta(0.1, 0.3)) ≈ -7.862370395825961
    @test mean(log, Beta(4.5, 0.3)) ≈ -0.07197681436958758
end

@testitem "Beta: mean(::typeof(mirrorlog))" begin
    include("distributions_setuptests.jl")

    @test mean(mirrorlog, Beta(1.0, 3.0)) ≈ -0.33333333333333337
    @test mean(mirrorlog, Beta(0.1, 0.3)) ≈ -0.9411396776150167
    @test mean(mirrorlog, Beta(4.5, 0.3)) ≈ -4.963371962929249
end

@testitem "Beta: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for a in 0.1:0.2:0.9, b in 0.1:0.2:0.9
        @testset let d = Beta(a, b)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)

            (η₁, η₂) = (a - 1, b - 1)

            for x in 0.1:0.1:0.9
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === oneunit(x)
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (log(x), log(1 - x)))
                @test @inferred(logpartition(ef)) ≈ (logbeta(η₁ + 1, η₂ + 1))
            end

            @test !@inferred(insupport(ef, -0.5))
            @test @inferred(insupport(ef, 0.5))

            # Not in the support
            @test_throws Exception logpdf(ef, -0.5)
        end
    end

    # Test failing isproper cases
    @test !isproper(MeanParametersSpace(), Beta, [-1])
    @test !isproper(MeanParametersSpace(), Beta, [1, -0.1])
    @test !isproper(MeanParametersSpace(), Beta, [-0.1, 1])
    @test !isproper(NaturalParametersSpace(), Beta, [-1.1])
    @test !isproper(NaturalParametersSpace(), Beta, [1, -1.1])
    @test !isproper(NaturalParametersSpace(), Beta, [-1.1, 1])

    # `a`s must add up to something more than 1, otherwise is not proper
    let ef = convert(ExponentialFamilyDistribution, Beta(0.1, 1.0))
        @test !isproper(prod(PreserveTypeProd(ExponentialFamilyDistribution), ef, ef))
    end

    # `b`s must add up to something more than 1, otherwise is not proper
    let ef = convert(ExponentialFamilyDistribution, Beta(1.0, 0.1))
        @test !isproper(prod(PreserveTypeProd(ExponentialFamilyDistribution), ef, ef))
    end
end

@testitem "Beta: prod with Distributions" begin
    include("distributions_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
        @test prod(strategy, Beta(3.0, 2.0), Beta(2.0, 1.0)) ≈ Beta(4.0, 2.0)
        @test prod(strategy, Beta(7.0, 1.0), Beta(0.1, 4.5)) ≈ Beta(6.1, 4.5)
        @test prod(strategy, Beta(1.0, 3.0), Beta(0.2, 0.4)) ≈ Beta(0.19999999999999996, 2.4)
    end

    @test @allocated(prod(ClosedProd(), Beta(3.0, 2.0), Beta(2.0, 1.0))) === 0
    @test @allocated(prod(GenericProd(), Beta(3.0, 2.0), Beta(2.0, 1.0))) === 0
end

@testitem "Beta: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for aleft in 0.51:1.0:5.0, aright in 0.51:1.0:5.0, bleft in 0.51:1.0:5.0, bright in 0.51:1.0:5.0
        @testset let (left, right) = (Beta(aleft, bleft), Beta(aright, bright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{Beta})
                )
            )
        end
    end
end
