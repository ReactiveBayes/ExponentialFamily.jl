module ChisqTest

using Test
using ExponentialFamily
using Random
using Distributions
using ForwardDiff
using StableRNGs
using ForwardDiff

import SpecialFunctions: logfactorial, loggamma
import ExponentialFamily:
    xtlog, ExponentialFamilyDistribution, getnaturalparameters, basemeasure,
    fisherinformation, sufficientstatistics

include("../testutils.jl")

@testset "Chisq" begin
    @testset "ExponentialFamilyDistribution{Chisq}" begin
        @testset for i in 3:7
            @testset let d = Chisq(2 * (i + 1))
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)
                η₁ = first(getnaturalparameters(ef))

                for x in (0.1,0.5, 1.0)
                    @test @inferred(isbasemeasureconstant(ef)) === NonConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === exp(-x/2)
                    @test @inferred(sufficientstatistics(ef, x)) === (log(x),)
                    @test @inferred(logpartition(ef)) ≈ loggamma(η₁ + 1) + (η₁+1)*log(2.0)
                end
            end
        end

        for space in (MeanParametersSpace(), NaturalParametersSpace())
            @test !isproper(space, Chisq, [Inf])
            @test !isproper(space, Chisq, [-1.0])
            @test !isproper(space, Chisq, [NaN])
            @test !isproper(space, Chisq, [1.0], NaN)
            @test !isproper(space, Chisq, [0.5, 0.5], 1.0)
        end
        ## mean parameter should be integer in the MeanParametersSpace
        @test !isproper(MeanParametersSpace(), Chisq, [0.1])
        @test_throws Exception convert(ExponentialFamilyDistribution, Chisq(Inf))
    end

    @testset "prod with Distribution and ExponentialFamilyDistribution" begin
        @testset for i=3:6
            left = Chisq(i+1)
            right = Chisq(i)
            prod_dist = prod(PreserveTypeProd(ExponentialFamilyDistribution),left, right)
            efleft = convert(ExponentialFamilyDistribution, left)
            efright = convert(ExponentialFamilyDistribution, right)
            prod_ef = prod(PreserveTypeProd(ExponentialFamilyDistribution),efleft,efright)
            η_left = getnaturalparameters(efleft)
            η_right = getnaturalparameters(efright)
            naturalparameters = η_left + η_right

            @test prod_dist.naturalparameters == naturalparameters
            @test getbasemeasure(prod_dist)(i) ≈ exp(-i)
            @test sufficientstatistics(prod_dist, i) === (log(i), )
            @test getlogpartition(prod_dist)(η_left + η_right) ≈ loggamma(η_left[1] + η_right[1] + 1)
            @test getsupport(prod_dist) === support(left)

            @test prod_ef.naturalparameters == naturalparameters
            @test getbasemeasure(prod_ef)(i) ≈ exp(-i)
            @test sufficientstatistics(prod_ef, i) === (log(i), )
            @test getlogpartition(prod_ef)(η_left + η_right) ≈ loggamma(η_left[1] + η_right[1] + 1)
            @test getsupport(prod_ef) === support(left)
        end
    end
end

end
