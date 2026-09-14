
# LogNormal comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "LogNormal: vague" begin
    include("distributions_setuptests.jl")

    @test LogNormal() == LogNormal(0.0, 1.0)
    @test typeof(vague(LogNormal)) <: LogNormal
    @test vague(LogNormal) == LogNormal(1, 1e12)
end

@testitem "LogNormal: prod with Distribution" begin
    include("distributions_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(LogNormal), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
        @test prod(strategy, LogNormal(1.0, 1.0), LogNormal(1.0, 1.0)) == LogNormal(0.5, sqrt(1 / 2))
        @test prod(strategy, LogNormal(2.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.5, sqrt(1 / 2))
        @test prod(strategy, LogNormal(1.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.0, sqrt(1 / 2))
        @test prod(strategy, LogNormal(1.0, 2.0), LogNormal(1.0, 2.0)) == LogNormal(-1.0, sqrt(2))
        @test prod(strategy, LogNormal(2.0, 2.0), LogNormal(2.0, 2.0)) == LogNormal(0.0, sqrt(2))
    end
end

@testitem "LogNormal: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for μleft in 10randn(4), μright in 10randn(4), σleft in 10rand(4), σright in 10rand(4)
        @testset let (left, right) = (LogNormal(μleft, σleft), LogNormal(μright, σright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{LogNormal})
                )
            )
        end
    end
end

@testitem "LogNormal: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for μ in 10randn(4), σ in 10rand(4)
        @testset let d = LogNormal(μ, σ)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)

            (η₁, η₂) = (μ / abs2(σ) - 1, -1 / (2abs2(σ)))

            for x in 10rand(4)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) ≈ invsqrt2π
                @test @inferred(sufficientstatistics(ef, x)) === (log(x), abs2(log(x)))
                @test @inferred(logpartition(ef)) ≈ (-(η₁ + 1)^2 / (4η₂) - 1 / 2 * log(-2η₂))
            end
        end
    end

    @test !isproper(DefaultParametersSpace(), LogNormal, [1.0])
    @test !isproper(DefaultParametersSpace(), LogNormal, [-1.0, 0.0])
    @test !isproper(DefaultParametersSpace(), LogNormal, [1.0, -1.0])
    @test !isproper(NaturalParametersSpace(), LogNormal, [1.0])
    @test !isproper(NaturalParametersSpace(), LogNormal, [-1.0, 0.0])
    @test !isproper(NaturalParametersSpace(), LogNormal, [1.0, 1.0])
end
