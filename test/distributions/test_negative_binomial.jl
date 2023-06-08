module NegativeBinomialTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff
using StableRNGs
import StatsFuns: logit
import DomainSets: NaturalNumbers
import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

@testset "NegativeBinomial" begin
    @testset "probvec" begin
        @test all(probvec(NegativeBinomial(2, 0.8)) .≈ (0.2, 0.8)) # check
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
        for nleft =1:15, pleft=0.01:0.3:0.99
            left = NegativeBinomial(nleft, pleft)
            efleft = convert(KnownExponentialFamilyDistribution, left)
            η_left = first(getnaturalparameters(efleft))
            for nright= 1:15, pright=0.01:0.3:0.99
                right = NegativeBinomial(nright, pright)
                efright = convert(KnownExponentialFamilyDistribution, right) 
                η_right = first(getnaturalparameters(efright))
                prod_dist = prod(ClosedProd(), left, right)
                prod_ef   = prod(efleft, efright)
                hist_sum(x) =
                prod_dist.basemeasure(x) * exp(
                    prod_dist.sufficientstatistics(x) * prod_dist.naturalparameters[1] -
                    prod_dist.logpartition(prod_dist.naturalparameters[1])
                )
                hist_sumef(x) =
                prod_ef.basemeasure(x) * exp(
                    prod_ef.sufficientstatistics(x) * prod_ef.naturalparameters[1] -
                    prod_ef.logpartition(prod_ef.naturalparameters[1])
                )
                @test sum(hist_sum(x) for x in 0:max(nleft,nright)) ≈ 1.0 atol=1e-5
                @test sum(hist_sumef(x) for x in 0:max(nleft, nright)) ≈ 1.0 atol=1e-5
                sample_points = collect(1:max(nleft,nright))
                for x in sample_points
                    @test prod_dist.basemeasure(x) == (binomial(BigInt(x + nleft - 1), x) * binomial(BigInt(x + nright - 1), x))
                    @test prod_dist.sufficientstatistics(x) == x
                    @test prod_ef.basemeasure(x) == (binomial(BigInt(x + nleft - 1), x) * binomial(BigInt(x + nright - 1), x))
                    @test prod_ef.sufficientstatistics(x) == x  
                end
            end
        end
    end

    @testset "natural parameters related" begin
        d1 = NegativeBinomial(5, 1 / 3)
        d2 = NegativeBinomial(5, 1 / 2)
        η1 = KnownExponentialFamilyDistribution(NegativeBinomial, [log(1 / 3)], 5)
        η2 = KnownExponentialFamilyDistribution(NegativeBinomial, [log(1 / 2)], 5)

        @test convert(KnownExponentialFamilyDistribution, d1) == η1
        @test convert(KnownExponentialFamilyDistribution, d2) == η2

        @test convert(Distribution, η1) ≈ d1
        @test convert(Distribution, η2) ≈ d2

        η3 = KnownExponentialFamilyDistribution(NegativeBinomial, [log(0.1)], 5)
        η4 = KnownExponentialFamilyDistribution(NegativeBinomial, [log(0.2)], 10)

        @test logpartition(η3) ≈ -5.0 * log(1 - 0.1)
        @test logpartition(η4) ≈ -10.0 * log(1 - 0.2)

        @test basemeasure(d1, 5) == binomial(9, 5)
        @test basemeasure(d2, 2) == binomial(6, 2)
        @test basemeasure(η1, 5) == basemeasure(d1, 5)
        @test basemeasure(η2, 2) == basemeasure(d2, 2)

        @test logpdf(η1, 2) == logpdf(d1, 2)
        @test logpdf(η2, 3) == logpdf(d2, 3)

        @test pdf(η1, 2) == pdf(d1, 2)
        @test pdf(η2, 4) == pdf(d2, 4)

        @test isproper(KnownExponentialFamilyDistribution(NegativeBinomial, [0], 5)) == true
        @test isproper(KnownExponentialFamilyDistribution(NegativeBinomial, [NaN], 5)) == false
        for x in 1:10
            ef_proper = KnownExponentialFamilyDistribution(NegativeBinomial, [-x], 5)
            ef_improper = KnownExponentialFamilyDistribution(NegativeBinomial, [x], 5)
            @test isproper(ef_proper) == true
            @test isproper(ef_improper) == false
        end
    end

    @testset "fisher information" begin
        for η in 1:10, r in 1:10
            ef = KnownExponentialFamilyDistribution(NegativeBinomial, [-η], r)
            f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(NegativeBinomial, η, r))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test fisherinformation(ef) ≈ autograd_information([-η])[1, 1]
        end

        rng = StableRNG(42)
        n_samples = 10000
        for η in 1:10, r in 1:10
            dist = NegativeBinomial(r, exp(-η))
            samples = rand(rng, dist, n_samples)
            hessian_at_sample =
                (sample) -> ForwardDiff.hessian((params) -> logpdf(NegativeBinomial(r, params[1]), sample), [exp(-η)])
            expected_hessian = -mean(hessian_at_sample, samples)
            # fisher information values are big, hard to compare directly.
            @test fisherinformation(NegativeBinomial(r, exp(-η))) / expected_hessian[1, 1] ≈ 1 atol = 0.01
        end
    end
end
end
