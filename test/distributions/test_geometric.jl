module GeometricTest

using Test
using Random
using Distributions
using ExponentialFamily
using StableRNGs
using ForwardDiff
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

include("../testutils.jl")

@testset "Geometric" begin
    @testset "ExponentialFamilyDistribution{Geometric}" begin
        @testset for p in 0.1:0.2:1.0
            @testset let d = Geometric(p)
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)
                η1 = first(getnaturalparameters(ef))

                for x in (1,3, 5)
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === one(x)
                    @test @inferred(sufficientstatistics(ef, x)) === (x,)
                    @test @inferred(logpartition(ef)) ≈  -log(one(η1) - exp(η1))
                end
            end
        end

        for space in (MeanParametersSpace(), NaturalParametersSpace())
            @test !isproper(space, Geometric, [2.0])
            @test !isproper(space, Geometric, [Inf])
            @test !isproper(space, Geometric, [NaN])
            @test !isproper(space, Geometric, [1.0], NaN)
            @test !isproper(space, Geometric, [0.5, 0.5], 1.0)
        end
        ## mean parameter should be integer in the MeanParametersSpace
        @test !isproper(MeanParametersSpace(), Geometric, [-0.1])
        @test_throws Exception convert(ExponentialFamilyDistribution, Geometric(Inf))
    end

    @testset "prod with Distributions" begin
        @test prod(ClosedProd(), Geometric(0.5), Geometric(0.6)) == Geometric(0.8)
        @test prod(ClosedProd(), Geometric(0.3), Geometric(0.8)) == Geometric(0.8600000000000001)
        @test prod(ClosedProd(), Geometric(0.5), Geometric(0.5)) == Geometric(0.75)

        @test @allocated(prod(ClosedProd(), Geometric(0.5), Geometric(0.6))) == 0
    end

    @testset "prod with ExponentialFamilyDistribution" for pleft in 0.1:0.2:0.5, pright in 0.5:0.2:1.0
        @testset let (left, right) = (Geometric(pleft), Geometric(pright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{Geometric})
                )
            )
        end
    end

end
end
