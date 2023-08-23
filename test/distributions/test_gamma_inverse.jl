module GammaInverseTest

using ExponentialFamily, Distributions
using Test, ForwardDiff, Random, StatsFuns, StableRNGs

include("../testutils.jl")

import SpecialFunctions: loggamma

# test this testset with $ make test testset='distributions:gamma_inverse'
@testset "GammaInverse" begin
    @testset "vague" begin
        d = vague(GammaInverse)
        @test typeof(d) <: GammaInverse
        @test mean(d) == huge
        @test params(d) == (2.0, huge)
    end

    # (α, θ) = (α_L + α_R + 1, θ_L + θ_R)
    @testset "prod" begin
        for strategy in (ClosedProd(), PreserveTypeProd(GammaInverse), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
            @test @inferred(prod(ClosedProd(), GammaInverse(3.0, 2.0), GammaInverse(2.0, 1.0))) ≈ GammaInverse(6.0, 3.0)
            @test @inferred(prod(ClosedProd(), GammaInverse(7.0, 1.0), GammaInverse(0.1, 4.5))) ≈ GammaInverse(8.1, 5.5)
            @test @inferred(prod(ClosedProd(), GammaInverse(1.0, 3.0), GammaInverse(0.2, 0.4))) ≈ GammaInverse(2.2, 3.4)
        end
    end

    # log(θ) - digamma(α)
    @testset "mean(::typeof(log))" begin
        @test mean(log, GammaInverse(1.0, 3.0)) ≈ 1.6758279535696414
        @test mean(log, GammaInverse(0.1, 0.3)) ≈ 9.21978213608514
        @test mean(log, GammaInverse(4.5, 0.3)) ≈ -2.5928437306854653
        @test mean(log, GammaInverse(42.0, 42.0)) ≈ 0.011952000346086233
    end

    # α / θ
    @testset "mean(::typeof(inv))" begin
        @test mean(inv, GammaInverse(1.0, 3.0)) ≈ 0.33333333333333333
        @test mean(inv, GammaInverse(0.1, 0.3)) ≈ 0.33333333333333337
        @test mean(inv, GammaInverse(4.5, 0.3)) ≈ 15.0000000000000000
        @test mean(inv, GammaInverse(42.0, 42.0)) ≈ 1.0000000000000000
    end

    @testset "ExponentialFamilyDistribution{InverseGamma}" begin
        @testset for α in 10rand(4), θ in 10rand(4)
            @testset let d = InverseGamma(α, θ)
                ef = test_exponentialfamily_interface(d; 
                    option_assume_no_allocations = true, 
                    test_fisherinformation_against_hessian = false,
                    test_fisherinformation_against_jacobian = false
                )

                run_test_fisherinformation_against_hessian(d; assume_no_allocations = false)
                run_test_fisherinformation_against_jacobian(d; assume_no_allocations = false)

                (α, β) = params(MeanParametersSpace(), d)

                (η₁, η₂) = (-α - 1, -β)

                for x in 10rand(4)
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === oneunit(x)
                    @test all(@inferred(sufficientstatistics(ef, x)) .≈ (log(x), inv(x)))
                    @test @inferred(logpartition(ef)) ≈ (loggamma(-η₁ - 1) - (-η₁ - 1)*log(-η₂))
                end

                @test !@inferred(insupport(ef, -0.5))
                @test @inferred(insupport(ef, 0.5))

                # Not in the support
                @test_throws Exception logpdf(ef, -0.5)
            end
        end

        # Test failing isproper cases
        @test !isproper(MeanParametersSpace(), InverseGamma, [-1])
        @test !isproper(MeanParametersSpace(), InverseGamma, [1, -0.1])
        @test !isproper(MeanParametersSpace(), InverseGamma, [-0.1, 1])
        @test !isproper(NaturalParametersSpace(), InverseGamma, [-0.5])
        @test !isproper(NaturalParametersSpace(), InverseGamma, [1, -1.1])
        @test !isproper(NaturalParametersSpace(), InverseGamma, [-0.5, 1])

    end

    @testset "prod with ExponentialFamilyDistribution" for aleft in 10rand(4), aright in 10rand(4), bleft in 10rand(4), bright in 10rand(4)
        @testset let (left, right) = (InverseGamma(aleft, bleft), InverseGamma(aright, bright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{InverseGamma})
                )
            )
        end
    end
end
end
