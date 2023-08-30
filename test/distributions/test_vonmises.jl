module VonMisesTest

using Test
using ExponentialFamily
using Distributions
using Random
using StableRNGs
using ForwardDiff
import SpecialFunctions: besseli

include("../testutils.jl")

@testset "VonMises" begin

    # VonMises comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(VonMises)

        @test typeof(d) <: VonMises
        @test mean(d) === 0.0
        @test params(d) === (0.0, 1.0e-12)
    end

    @testset "ExponentialFamilyDistribution{VonMises}" begin
        @testset for a in -2:1.0:2, b in 0.1:4.0:10.0
            @testset let d = VonMises(a, b)
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false)

                for x in a-1:0.5:a+1
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
        @test !isproper(MeanParametersSpace(), VonMises, [-1])
        @test !isproper(MeanParametersSpace(), VonMises, [1], 3.0)
        @test !isproper(MeanParametersSpace(), VonMises, [1, -2])
    end

    @testset "prod with Distributions" begin
        function prod_result_parameters(paramsleft, paramsright)
            (μleft, κleft) = paramsleft
            (μright, κright) = paramsright
            a = κleft * cos(μleft) + κright * cos(μright)
            b = κleft * sin(μleft) + κright * sin(μright)

            R = sqrt(a^2 + b^2)
            α = atan(b / a)

            return α, R
        end
        (μ1, κ1) = prod_result_parameters((3.0, 2.0), (2.0, 1.0))
        (μ2, κ2) = prod_result_parameters((7.0, 1.0), (0.1, 4.5))
        (μ3, κ3) = prod_result_parameters((1.0, 3.0), (0.2, 0.4))
        for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
            @test prod(strategy, VonMises(3.0, 2.0), VonMises(2.0, 1.0)) ≈ VonMises(μ1, κ1)
            @test prod(strategy, VonMises(7.0, 1.0), VonMises(0.1, 4.5)) ≈ VonMises(μ2, κ2)
            @test prod(strategy, VonMises(1.0, 3.0), VonMises(0.2, 0.4)) ≈ VonMises(μ3, κ3)
        end
    end

    @testset "prod with ExponentialFamilyDistribution{VonMises}" begin
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
end

end
