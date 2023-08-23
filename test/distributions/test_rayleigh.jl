module RayleighTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff
using DomainSets
using StableRNGs
using HCubature

import ExponentialFamily: mirrorlog, ExponentialFamilyDistribution, logpartition,
    basemeasure, getbasemeasure, getnaturalparameters, getsufficientstatistics, fisherinformation, support, getsupport

include("../testutils.jl")

@testset "Rayleigh" begin
    @testset "ExponentialFamilyDistribution{Rayleigh}" begin
        @testset for σ in 3
            @testset let d = Rayleigh(σ)
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)
                η1 = first(getnaturalparameters(ef))

                for x in (0.1,0.5, 1.0)
                    @test @inferred(isbasemeasureconstant(ef)) === NonConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === x
                    @test @inferred(sufficientstatistics(ef, x)) === (x^2,)
                    @test @inferred(logpartition(ef)) ≈ -log(-2 * η1)
                end
            end
        end

        for space in (MeanParametersSpace(), NaturalParametersSpace())
            @test !isproper(space, Rayleigh, [Inf])
           
            @test !isproper(space, Rayleigh, [NaN])
            @test !isproper(space, Rayleigh, [1.0], NaN)
            @test !isproper(space, Rayleigh, [0.5, 0.5], 1.0)
        end
        @test !isproper(MeanParametersSpace(), Rayleigh, [-1.0])
        @test_throws Exception convert(ExponentialFamilyDistribution, Rayleigh(Inf))
    end


    @testset "prod with PreserveTypeProd{ExponentialFamilyDistribution}" for σleft in 1:4, σright in 4:7
        @testset let (left,right) = (Rayleigh(σleft), Rayleigh(σright))
            ef_left = convert(ExponentialFamilyDistribution, left)
            ef_right = convert(ExponentialFamilyDistribution, right)
            prod_dist = prod(PreserveTypeProd(ExponentialFamilyDistribution),left,right)
            @test first(hquadrature(x -> pdf(prod_dist,tan(x* pi /2))*(pi/2)*(1 / cos(x * pi / 2)^2), 0.0, 1.0)) ≈ 1.0   
            @test getnaturalparameters(prod_dist) == getnaturalparameters(ef_left) + getnaturalparameters(ef_right)
        end
    end

end

end
