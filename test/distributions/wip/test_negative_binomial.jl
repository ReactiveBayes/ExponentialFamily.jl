module NegativeBinomialTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff
using StableRNGs
import StatsFuns: logit
import DomainSets: NaturalNumbers
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

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

    @testset "prod" begin
        for nleft in 1:15, pleft in 0.01:0.3:0.99
            left = NegativeBinomial(nleft, pleft)
            efleft = convert(ExponentialFamilyDistribution, left)
            η_left = getnaturalparameters(efleft)
            for nright in 1:15, pright in 0.01:0.3:0.99
                right = NegativeBinomial(nright, pright)
                efright = convert(ExponentialFamilyDistribution, right)
                η_right = first(getnaturalparameters(efright))
                prod_dist = prod(ClosedProd(), left, right)
                prod_ef = prod(efleft, efright)
                hist_sum(x) =
                    prod_dist.basemeasure(x) * exp(
                        prod_dist.sufficientstatistics(x)' * prod_dist.naturalparameters -
                        prod_dist.logpartition(prod_dist.naturalparameters)
                    )
                hist_sumef(x) =
                    prod_ef.basemeasure(x) * exp(
                        prod_ef.sufficientstatistics(x)' * prod_ef.naturalparameters -
                        prod_ef.logpartition(prod_ef.naturalparameters)
                    )
                @test sum(hist_sum(x) for x in 0:max(nleft, nright)) ≈ 1.0 atol = 1e-5
                @test sum(hist_sumef(x) for x in 0:max(nleft, nright)) ≈ 1.0 atol = 1e-5
                sample_points = collect(1:max(nleft, nright))
                for x in sample_points
                    @test prod_dist.basemeasure(x) ==
                          (binomial(BigInt(x + nleft - 1), x) * binomial(BigInt(x + nright - 1), x))
                    @test prod_dist.sufficientstatistics(x) == [x]
                    @test prod_ef.basemeasure(x) ==
                          (binomial(BigInt(x + nleft - 1), x) * binomial(BigInt(x + nright - 1), x))
                    @test prod_ef.sufficientstatistics(x) == [x]
                end
            end
        end
    end

    @testset "natural parameters related" begin
        for r in 1:5
            for p in 0.1:0.1:0.9
                dist = NegativeBinomial(r, p)
                ef_manual = ExponentialFamilyDistribution(NegativeBinomial, [log(one(Float64) - p)], r)
                ef_converted = convert(ExponentialFamilyDistribution, dist)

                @test ef_manual == ef_converted
                @test convert(Distribution, ef_manual) ≈ dist
                @test logpartition(ef_converted) ≈ -r * log(p)
                for k in 3:5
                    @test logpdf(dist, k) ≈ logpdf(ef_manual, k)
                    @test pdf(dist, k) ≈ pdf(ef_manual, k)
                    @test basemeasure(ef_converted, k) == binomial(Int(k + r - 1), k)
                end
                @test sum(pdf(ef_converted, x) for x in 0:200) ≈ 1.0 atol = 1e-5
            end
        end
    end
    @testset "Proper" begin
        for x in 1:10
            ef_proper = ExponentialFamilyDistribution(NegativeBinomial, [-x], 5)
            ef_improper = ExponentialFamilyDistribution(NegativeBinomial, [x], 5)
            @test isproper(ef_proper) == true
            @test isproper(ef_improper) == false
        end
    end
    transformation(η) = 1 - exp(η[1])
    @testset "fisher information" begin
        for η in 1:10, r in 1:10
            ef = ExponentialFamilyDistribution(NegativeBinomial, [-η], r)
            dist = convert(Distribution, ef)
            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(NegativeBinomial, η, r))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            J = ForwardDiff.gradient(transformation, [-η])
            fef = fisherinformation(ef)
            fdist = fisherinformation(dist)
            @test first(fef) ≈ first(autograd_information([-η]))
            @test J' * fdist * J ≈ first(fef)
        end
    end

    @testset "ExponentialFamilyDistribution mean,var" begin
        for η in 1:4, r in 1:4
            dist = NegativeBinomial(r, exp(-η))
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end
end
