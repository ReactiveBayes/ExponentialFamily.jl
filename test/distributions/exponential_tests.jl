
# Exponential comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Exponential: vague" begin
    include("distributions_setuptests.jl")

    d = vague(Exponential)

    @test typeof(d) <: Exponential
    @test mean(d) === 1e12
    @test params(d) === (1e12,)
end

@testitem "Exponential: mean(::typeof(log))" begin
    include("distributions_setuptests.jl")

    @test mean(log, Exponential(1)) ≈ -MathConstants.eulergamma
    @test mean(log, Exponential(10)) ≈ 1.7253694280925127
    @test mean(log, Exponential(0.1)) ≈ -2.8798007578955787
end

@testitem "Exponential: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for scale in (0.1, 1.0, 10.0, 10.0rand())
        @testset let d = Exponential(scale)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)

            (η₁,) = -inv(scale)

            for x in [100rand() for _ in 1:4]
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === oneunit(x)
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x,))
                @test @inferred(logpartition(ef)) ≈ (-log(-η₁))
            end

            @test !@inferred(insupport(ef, -0.5))
            @test @inferred(insupport(ef, 0.5))

            # Not in the support
            @test_throws Exception logpdf(ef, -0.5)
        end
    end

    # Test failing isproper cases
    @test !isproper(MeanParametersSpace(), Exponential, [-1])
    @test !isproper(MeanParametersSpace(), Exponential, [1, -0.1])
    @test !isproper(MeanParametersSpace(), Exponential, [-0.1, 1])
    @test !isproper(NaturalParametersSpace(), Exponential, [1.1])
    @test !isproper(NaturalParametersSpace(), Exponential, [1, -1.1])
    @test !isproper(NaturalParametersSpace(), Exponential, [-1.1, 1])
end

@testitem "Exponential: prod with Distributiond" begin
    include("distributions_setuptests.jl")

    for strategy in (GenericProd(), ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd())
        @test prod(strategy, Exponential(5), Exponential(4)) ≈ Exponential(1 / 0.45)
        @test prod(strategy, Exponential(1), Exponential(1)) ≈ Exponential(1 / 2)
        @test prod(strategy, Exponential(0.1), Exponential(0.1)) ≈ Exponential(0.05)
    end
end

@testitem "Exponential: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for sleft in (0.1, 1.0, 10.0, 10.0rand()), sright in (0.1, 1.0, 10.0, 10.0rand())
        @testset let (left, right) = (Exponential(sleft), Exponential(sright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{Exponential})
                )
            )
        end
    end
end
