module LogNormalTest

using ExponentialFamily, Distributions
using Test, ForwardDiff, Random, StatsFuns, StableRNGs

include("../testutils.jl")

@testset "LogNormal" begin
    @testset "constructors" begin
        @test LogNormal() == LogNormal(0.0, 1.0)
        @test typeof(vague(LogNormal)) <: LogNormal
        @test vague(LogNormal) == LogNormal(1, 1e12)
    end

    @testset "prod with Distribution" begin
        for strategy in (ClosedProd(), PreserveTypeProd(LogNormal), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
            @test prod(strategy, LogNormal(1.0, 1.0), LogNormal(1.0, 1.0)) == LogNormal(0.5, sqrt(1 / 2))
            @test prod(strategy, LogNormal(2.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.5, sqrt(1 / 2))
            @test prod(strategy, LogNormal(1.0, 1.0), LogNormal(2.0, 1.0)) == LogNormal(1.0, sqrt(1 / 2))
            @test prod(strategy, LogNormal(1.0, 2.0), LogNormal(1.0, 2.0)) == LogNormal(-1.0, sqrt(2))
            @test prod(strategy, LogNormal(2.0, 2.0), LogNormal(2.0, 2.0)) == LogNormal(0.0, sqrt(2))
        end
    end

    @testset "prod with ExponentialFamilyDistribution" begin
        for μleft in 10randn(4), μright in 10randn(4), σleft in 10rand(4), σright in 10rand(4)
            let left = LogNormal(μleft, σleft), right = LogNormal(μright, σright)
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

    @testset "ExponentialFamilyDistribution{LogNormal}" begin
        @testset for μ in 10randn(4), σ in 10rand(4)
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

        @test !isproper(MeanParametersSpace(), LogNormal, [1.0])
        @test !isproper(MeanParametersSpace(), LogNormal, [-1.0, 0.0])
        @test !isproper(MeanParametersSpace(), LogNormal, [1.0, -1.0])
        @test !isproper(NaturalParametersSpace(), LogNormal, [1.0])
        @test !isproper(NaturalParametersSpace(), LogNormal, [-1.0, 0.0])
        @test !isproper(NaturalParametersSpace(), LogNormal, [1.0, 1.0])
    end
end
end
