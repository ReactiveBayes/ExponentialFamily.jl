module MvNormalWishartTests

using Distributions, HCubature
using Test
using StatsFuns: logmvgamma
using LinearAlgebra
using ForwardDiff
using ExponentialFamily
import ExponentialFamily:
    MvNormalWishart, ExponentialFamilyDistribution, params, dof, invscatter
import ExponentialFamily:
    scale, dim, getnaturalparameters, tiny, logpartition, cholinv, MvNormalMeanPrecision, sufficientstatistics
using Distributions
using Random

include("../testutils.jl")


@testset "MvNormalWishart" begin
    @testset "common" begin
        m = rand(2)
        dist = MvNormalWishart(m, [1.0 0.0; 0.0 1.0], 0.1, 3.0)
        @test params(dist) == (m, [1.0 0.0; 0.0 1.0], 0.1, 3.0)
        @test dof(dist) == 3.0
        @test invscatter(dist) == [1.0 0.0; 0.0 1.0]
        @test scale(dist) == 0.1
        @test dim(dist) == 2
    end

    @testset "ExponentialFamilyDistribution{MvNormalWishart}" begin
        @testset for dim in (3), invS in rand(Wishart(10,diageye(dim)),2)
            ν = dim + 2
            @testset let (d = MvNormalWishart(rand(dim), invS, rand(),ν))
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false, test_fisherinformation_properties = false,
                test_fisherinformation_against_hessian = false,
                test_fisherinformation_against_jacobian = false,)
                (η1,η2) = unpack_parameters(MvNormalWishart,getnaturalparameters(ef))
          
                # for x in diageye(dim)
                #     @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                #     @test @inferred(basemeasure(ef, x)) === 1.0
                #     @test @inferred(sufficientstatistics(ef, x)) === (logdet(x), x)
                #     @test @inferred(logpartition(ef)) ≈ -(η1 + (dim + 1) / 2) * logdet(-η2) + logmvgamma(dim, η1 + (dim + 1) / 2)
                # end
            end
        end
    end



    # @testset "conversions" begin
    #     for i in 1:10, j in 2:6
    #         m = rand(j)
    #         κ = rand()
    #         Ψ = diagm(rand(j))
    #         ν = 2 * j + 1
    #         dist = MvNormalWishart(m, Ψ, κ, ν)
    #         ef = convert(ExponentialFamilyDistribution, dist)

    #         @test getnaturalparameters(ef) ≈ vcat(κ * m, vec(-(1 / 2) * (inv(Ψ) + κ * m * m')), -κ / 2, (ν - j) / 2)
    #         @test invscatter(convert(Distribution, ef)) ≈ cholinv(Ψ)
    #         @test dof(convert(Distribution, ef)) == 2 * j + 1
    #     end
    # end

    # @testset "exponential family functions" begin
    #     for i in 1:10, j in 2:5, κ in 0.01:1.0:5.0
    #         m = rand(j)
    #         Ψ = m * m' + I
    #         dist = MvNormalWishart(m, Ψ, κ, j + 1)
    #         ef = convert(ExponentialFamilyDistribution, dist)
    #         @test pdf(dist, [m, Ψ]) ≈ normal_wishart_pdf(m, Ψ, m, κ, float(j + 1), Ψ)
    #         @test logpdf(dist, [m, Ψ]) ≈ log(normal_wishart_pdf(m, Ψ, m, κ, float(j + 1), Ψ))
    #     end
    # end

    # @testset "sampling" begin
    #     nsamples = 100
    #     for j in 2:5, κ in 1:5
    #         m = rand(j)
    #         Ψ = m * m' + I
    #         dist = MvNormalWishart(m, Ψ, κ, j + 3)
    #         ef = convert(ExponentialFamilyDistribution, dist)
    #         st = sufficientstatistics(dist)
    #         samples = rand(MersenneTwister(j), dist, nsamples)
    #         ηvec = getnaturalparameters(ef)
    #         expsuffstats = sum(st(sample[1], sample[2]) for sample in samples) / nsamples
    #         expsuffstatsvec = ForwardDiff.gradient(x -> logpartition(ef, x), ηvec)
    #         @test expsuffstats ≈ expsuffstatsvec rtol = 0.1
    #     end
    # end

    # @testset "prod" begin
    #     for j in 2:2, κ in 1:2
    #         m1 = rand(j)
    #         m2 = rand(j)
    #         Ψ1 = m1 * m1' + I
    #         Ψ2 = m2 * m2' + I
    #         dist1 = MvNormalWishart(m1, Ψ1, κ, j + 3)
    #         dist2 = MvNormalWishart(m2, Ψ2, κ, j + 3)
    #         ef1 = convert(ExponentialFamilyDistribution, dist1)
    #         ef2 = convert(ExponentialFamilyDistribution, dist2)
    #         @test prod(ClosedProd(), dist1, dist2) == convert(Distribution, prod(ClosedProd(), ef1, ef2))
    #     end
    # end
end

end
