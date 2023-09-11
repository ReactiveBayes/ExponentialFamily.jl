module NegativeBinomialTest

using Test
using ExponentialFamily
using Distributions
import ExponentialFamily: ExponentialFamilyDistribution,getnaturalparameters

include("../testutils.jl")

@testset "NegativeBinomial" begin
    @testset "probvec" begin
        @test all(probvec(NegativeBinomial(2, 0.8)) .≈ (0.2, 0.8))
        @test probvec(NegativeBinomial(2, 0.2)) == (0.8, 0.2)
        @test probvec(NegativeBinomial(2, 0.1)) == (0.9, 0.1)
        @test probvec(NegativeBinomial(2)) == (0.5, 0.5)
    end

    @testset "vague" begin
        @test_throws MethodError vague(NegativeBinomial)
        @test_throws MethodError vague(NegativeBinomial, 1 / 2)

        vague_dist = vague(NegativeBinomial, 5)
        @test typeof(vague_dist) <: NegativeBinomial
        @test probvec(vague_dist) == (0.5, 0.5)
    end

    @testset "ExponentialFamilyDistribution{NegativeBinomial}" begin
        @testset for p in (0.1,0.4), r in (2,3,4)
            @testset let d = NegativeBinomial(r, p)
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false)
                for x in 2:4
                    @test @inferred(isbasemeasureconstant(ef)) === NonConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === binomial(Int(x + r - 1), x)
                    @test @inferred(sufficientstatistics(ef, x)) === (x,)
                    @test @inferred(logpartition(ef)) ≈ -r*log(1 - exp(getnaturalparameters(ef)[1]))
                end
            end
        end

        for space in (MeanParametersSpace(), NaturalParametersSpace())
            @test !isproper(space, NegativeBinomial, [Inf], 1.0)
            @test !isproper(space, NegativeBinomial, [1.0], Inf)
            @test !isproper(space, NegativeBinomial, [NaN], 1.0)
            @test !isproper(space, NegativeBinomial, [1.0], NaN)
            @test !isproper(space, NegativeBinomial, [0.5, 0.5], 1.0)

            # Conditioner is required
            @test_throws Exception isproper(space, NegativeBinomial, [0.5], [0.5, 0.5])
            @test_throws Exception isproper(space, NegativeBinomial, [1.0], nothing)
            @test_throws Exception isproper(space, NegativeBinomial, [1.0], nothing)
        end

        @test_throws Exception convert(ExponentialFamilyDistribution, NegativeBinomial(Inf, Inf))
    end

    @testset "prod" begin
        for nleft in 3:5, pleft in 0.01:0.3:0.99
            left = NegativeBinomial(nleft, pleft)
            efleft = convert(ExponentialFamilyDistribution, left)
            η_left = getnaturalparameters(efleft)
            for nright in 6:7, pright in 0.01:0.3:0.99
                right = NegativeBinomial(nright, pright)
                efright = convert(ExponentialFamilyDistribution, right)
                η_right = first(getnaturalparameters(efright))
                prod_dist = prod(PreserveTypeProd(ExponentialFamilyDistribution), left, right)
           
                @test sum(pdf(prod_dist, x) for x in 0:max(nleft, nright)) ≈ 1.0 atol = 1e-5
            end
        end
    end


end
end
