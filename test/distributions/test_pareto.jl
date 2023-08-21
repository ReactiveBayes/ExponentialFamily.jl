module ParetoTest

using Test
using Random
using Distributions
using ExponentialFamily
using ForwardDiff
using StableRNGs
using HCubature 
import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation, getconditioner

include("../testutils.jl")

@testset "Pareto" begin
    @testset "ExponentialFamilyDistribution{Pareto}" begin
        @testset for shape in (1.0, 2.0, 3.0), scale in (0.25, 0.5, 2.0)
            @testset let d = Pareto(shape, scale)
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false)
                η1 = -shape - 1 
                for x in scale:1.0:scale + 3.0
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === oneunit(x)
                    @test @inferred(sufficientstatistics(ef, x)) === (log(x),)
                    @test @inferred(logpartition(ef)) ≈ log(scale^(one(η1) + η1) / (-one(η1) - η1))
                end
            end
        end

        for space in (MeanParametersSpace(), NaturalParametersSpace())
            @test !isproper(space, Pareto, [Inf], 1.0)
            @test !isproper(space, Pareto, [1.0], Inf)
            @test !isproper(space, Pareto, [NaN], 1.0)
            @test !isproper(space, Pareto, [1.0], NaN)
            @test !isproper(space, Pareto, [0.5, 0.5], 1.0)

            # Conditioner is required
            @test_throws Exception isproper(space, Pareto, [0.5], [0.5, 0.5])
            @test_throws Exception isproper(space, Pareto, [1.0], nothing)
            @test_throws Exception isproper(space, Pareto, [1.0], nothing)
        end

        @test_throws Exception convert(ExponentialFamilyDistribution, Pareto(Inf, Inf))
    end

    @testset "prod with Distributions" begin
        @test prod(PreserveTypeProd(Pareto), Pareto(0.5), Pareto(0.6)) == Pareto(2.1)
        @test prod(PreserveTypeProd(Pareto), Pareto(0.3), Pareto(0.8)) == Pareto(2.1)
        @test prod(PreserveTypeProd(Pareto), Pareto(0.5), Pareto(0.5)) == Pareto(2.0)
        @test prod(PreserveTypeProd(Pareto), Pareto(3), Pareto(2)) == Pareto(6.0)
    end

    @testset "prod with ExponentialFamilyDistribution{Pareto}" begin
        for conditioner in (0.01, 1.0), alphaleft in 0.1:0.1:0.9, alpharight in 0.1:0.1:0.9
            let left = Pareto(alphaleft,conditioner), right = Pareto(alpharight,conditioner)
                @test test_generic_simple_exponentialfamily_product(
                    left,
                    right,
                    strategies = (PreserveTypeProd(ExponentialFamilyDistribution{Pareto}), GenericProd())
                )
            end
        end

        # Different conditioner parameters cannot be compute a closed prod with the same type
        @test_throws Exception prod(
            PreserveTypeProd(ExponentialFamilyDistribution{Pareto}),
            convert(ExponentialFamilyDistribution, Pareto(0.0, 0.54)),
            convert(ExponentialFamilyDistribution, Pareto(0.01, 0.5))
        )
        @test_throws Exception prod(
            PreserveTypeProd(ExponentialFamilyDistribution{Pareto}),
            convert(ExponentialFamilyDistribution, Pareto(1.0, 0.56)),
            convert(ExponentialFamilyDistribution, Pareto(2.0, 0.5))
        )
    end

    @testset "prod with different conditioner" begin
        for conditioner_left in (2,3), conditioner_right in (4, 5), alphaleft in 0.1:0.1:0.3, alpharight in 0.1:0.1:0.3
            let left = Pareto(alphaleft,conditioner_left), right = Pareto(alpharight,conditioner_right)
                ef_left = convert(ExponentialFamilyDistribution, left)
                ef_right = convert(ExponentialFamilyDistribution, right)
                prod_dist = prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_left,ef_right)
                @test getnaturalparameters(prod_dist) ≈ getnaturalparameters(ef_left) + getnaturalparameters(ef_right)
                @test getsupport(prod_dist).lb == max(conditioner_left,conditioner_right)
                @test sufficientstatistics(prod_dist, (max(conditioner_left,conditioner_right)+1)) === (log(max(conditioner_left,conditioner_right)+1), )
                @test first(hquadrature(x -> pdf(prod_dist,tan(x* pi /2))*(pi/2)*(1 / cos(x * pi / 2)^2),(2/pi)*atan(getsupport(prod_dist).lb), 1.0)) ≈ 1.0               
            end
        end
    end
end
end
