
module GammaFamilyTest

using ExponentialFamily, Distributions
using Test, Random, ForwardDiff, StableRNGs

import SpecialFunctions: loggamma
import ExponentialFamily: xtlog

include("../../testutils.jl")

@testset "Gamma family" begin

    @testset "Base statistical methods" begin 

        compare_basic_statistics = (left, right) -> begin
            @test mean(left) ≈ mean(right)
            @test var(left) ≈ var(right)
            @test cov(left) ≈ cov(right)
            @test shape(left) ≈ shape(right)
            @test scale(left) ≈ scale(right)
            @test rate(left) ≈ rate(right)
            @test entropy(left) ≈ entropy(right)
            @test pdf(left, 1.0) ≈ pdf(right, 1.0)
            @test pdf(left, 10.0) ≈ pdf(right, 10.0)
            @test logpdf(left, 1.0) ≈ logpdf(right, 1.0)
            @test logpdf(left, 10.0) ≈ logpdf(right, 10.0)
        
            @test mean(log, left) ≈ mean(log, right)
            @test mean(loggamma, left) ≈ mean(loggamma, right)
            @test mean(xtlog, left) ≈ mean(xtlog, right)

            return true
        end

        types = ExponentialFamily.union_types(GammaDistributionsFamily{Float64})
        rng   = MersenneTwister(1234)
        for _ in 1:10
            for type in types
                left = convert(type, 100 * rand(rng, Float64), 100 * rand(rng, Float64))
                for type in types
                    right = convert(type, left)
                    @test compare_basic_statistics(left, right)

                    @test all(params(MeanParametersSpace(), left) .== (shape(left), scale(left)))
                    @test all(params(MeanParametersSpace(), right) .== (shape(right), scale(right)))
                end
            end
        end
    end

    @testset "ExponentialFamilyDistribution{Gamma}" begin
        @testset for k in (0.1, 2.0, 5.0), θ in (0.1, 2.0, 5.0), T in ExponentialFamily.union_types(GammaDistributionsFamily{Float64})
            @testset let d = convert(T, GammaShapeScale(k, θ))
                ef = test_exponentialfamily_interface(d)

                (η₁, η₂) = (shape(d) - 1, -inv(scale(d)))

                for x in 0.1:0.5:5.0
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === oneunit(x)
                    @test all(@inferred(sufficientstatistics(ef, x)) .=== (log(x), x))
                    @test @inferred(logpartition(ef)) ≈ loggamma(η₁ + 1) - (η₁ + 1) * log(-η₂)
                end

                @test @inferred(insupport(ef, 0.5))
                @test !@inferred(insupport(ef, -0.5))

                # # Not in the support
                @test_throws Exception logpdf(ef, -0.5)
            end
        end

        # Test failing isproper cases
        @test !isproper(MeanParametersSpace(), Gamma, [-1])
        @test !isproper(MeanParametersSpace(), Gamma, [1, -1])
        @test !isproper(MeanParametersSpace(), Gamma, [-1, -1])
        @test !isproper(NaturalParametersSpace(), Gamma, [-1])
        @test !isproper(NaturalParametersSpace(), Gamma, [1, 10])
        @test !isproper(NaturalParametersSpace(), Gamma, [-100, -1])
    end

    @testset "prod with ExponentialFamilyDistribution" for kleft in 0.51:1.0:5.0, kright in 0.51:1.0:5.0, θleft in 0.1:1.0:5.0, θright in 0.1:1.0:5.0,
        Tleft in ExponentialFamily.union_types(GammaDistributionsFamily{Float64}), Tright in ExponentialFamily.union_types(GammaDistributionsFamily{Float64})
        @testset let (left, right) = (convert(Tleft, Gamma(kleft, θleft)), convert(Tright, Gamma(kright, θright)))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{Gamma})
                )
            )
        end

    end

    # shapes must add up to something more than 1, otherwise is not proper
    let ef = convert(ExponentialFamilyDistribution, Gamma(0.1, 1.0))
        @test !isproper(prod(PreserveTypeProd(ExponentialFamilyDistribution), ef, ef))
    end
end

end
