module ContingencyTest

using Test
using ExponentialFamily
using Distributions
using Random
using StatsFuns
import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure
import Distributions: cdf

@testset "Contingency" begin
    @testset "common" begin
        @test Contingency <: Distribution
        @test Contingency <: DiscreteDistribution
        @test Contingency <: MultivariateDistribution

        @test value_support(Contingency) === Discrete
        @test variate_form(Contingency) === Multivariate
    end

    @testset "contingency_matrix" begin
        @test ExponentialFamily.contingency_matrix(Contingency(ones(3, 3))) == ones(3, 3) ./ 9
        @test ExponentialFamily.contingency_matrix(Contingency(ones(3, 3), Val(true))) == ones(3, 3) ./ 9
        @test ExponentialFamily.contingency_matrix(Contingency(ones(3, 3), Val(false))) == ones(3, 3) # Matrix is wrong, but just to test that `false` is working
        @test ExponentialFamily.contingency_matrix(Contingency(ones(4, 4))) == ones(4, 4) ./ 16
        @test ExponentialFamily.contingency_matrix(Contingency(ones(4, 4), Val(true))) == ones(4, 4) ./ 16
        @test ExponentialFamily.contingency_matrix(Contingency(ones(4, 4), Val(false))) == ones(4, 4)
    end

    @testset "vague" begin
        @test_throws MethodError vague(Contingency)

        d1 = vague(Contingency, 3)

        @test typeof(d1) <: Contingency
        @test ExponentialFamily.contingency_matrix(d1) ≈ ones(3, 3) ./ 9

        d2 = vague(Contingency, 4)

        @test typeof(d2) <: Contingency
        @test ExponentialFamily.contingency_matrix(d2) ≈ ones(4, 4) ./ 16
    end

    @testset "pdf" begin
        d1 = vague(Contingency, 3)
        d2 = Contingency(ones(3, 3), Val(true))
        d3 = vague(Contingency, 2)
        @test_throws MethodError pdf(d1, 1)
        @test_throws AssertionError pdf(d1, [1, 2, 3, 4])
        @test_throws AssertionError pdf(d1, [1.1])
        @test pdf(d1, [1, 2]) == ExponentialFamily.contingency_matrix(d1)[1, 2]
        @test pdf(d1, [true false false; false true false]) == pdf(d1, [1, 2])
        @test logpdf(d1, [1, 2]) == log(ExponentialFamily.contingency_matrix(d1)[1, 2])
        @test logpdf(d1, [true false false; false true false]) == logpdf(d1, [1, 2])
        @test logpdf(d2, [2, 3]) == log(ExponentialFamily.contingency_matrix(d2)[2, 3])
        @test mean(d3) == [3 / 2, 3 / 2]

        @test cov(d3) == [0.25 0; 0 0.25]
        @test var(d3) == [0.25, 0.25]
    end

    @testset "natural parameters related" begin
        d1           = vague(Contingency, 2)
        d2           = vague(Contingency, 2)
        ηcontingency = KnownExponentialFamilyDistribution(Contingency, log.([0.1/0.15 0.7/0.15; 0.05/0.15 1.0]))
        @test getnaturalparameters(ηcontingency) == log.([0.1/0.15 0.7/0.15; 0.05/0.15 1.0])
        @test convert(KnownExponentialFamilyDistribution, Contingency([0.1 0.7; 0.05 0.15])) ==
              KnownExponentialFamilyDistribution(Contingency, log.([0.1/0.15 0.7/0.15; 0.05/0.15 1.0]))
        @test d1 == d2
        @test convert(KnownExponentialFamilyDistribution, d1) ==
              KnownExponentialFamilyDistribution(Contingency, log.([1.0 1.0; 1.0 1.0]))
        @test convert(Distribution, ηcontingency) ≈ Contingency([0.1 0.7; 0.05 0.15])
        @test prod(ηcontingency, ηcontingency) ==
              KnownExponentialFamilyDistribution(Contingency, 2log.([0.1/0.15 0.7/0.15; 0.05/0.15 1.0]))

        @test basemeasure(d1, rand()) == 1.0
        @test basemeasure(d2, [1, 2]) == 1.0

        @test logpartition(ηcontingency) == log(0.1 / 0.15 + 0.7 / 0.15 + 0.05 / 0.15 + 1.0)
    end

    @testset "entropy" begin
        @test entropy(Contingency([0.7 0.1; 0.1 0.1])) ≈ 0.9404479886553263
        @test entropy(Contingency(10.0 * [0.7 0.1; 0.1 0.1])) ≈ 0.9404479886553263
        @test entropy(Contingency([0.07 0.41; 0.31 0.21])) ≈ 1.242506182893139
        @test entropy(Contingency(10.0 * [0.07 0.41; 0.31 0.21])) ≈ 1.242506182893139
        @test entropy(Contingency([0.09 0.00; 0.00 0.91])) ≈ 0.30253782309749805
        @test entropy(Contingency(10.0 * [0.09 0.00; 0.00 0.91])) ≈ 0.30253782309749805
        @test !isnan(entropy(Contingency([0.0 1.0; 1.0 0.0])))
        @test !isinf(entropy(Contingency([0.0 1.0; 1.0 0.0])))
    end
    @testset "cdf" begin
        dist1 = Contingency([0.2 0.3; 0.4 0.1])
        dist2 = Contingency(softmax(rand(3, 3)))
        @test cdf(dist1, [1.0, 2.0]) ≈ 0.5
        @test cdf(dist2, [1, 3]) == sum(dist2.p[1, 1:3])
        @test cdf(dist2, [0.5, 0.5]) == 0.0
        @test cdf(dist2, [0.5, 0.5]) == 0.0
        @test cdf(dist2, [3, 3]) == 1.0
        @test cdf(dist2, [0, 0]) == 0.0
        @test cdf(dist2, [3 / 2, 2]) == cdf(dist2, [1, 2])
        @test cdf(dist1, [3.2, 0.1]) == 0.0
        @test cdf(dist1, [3, 6]) == 1.0

        @test icdf(dist2, 0.0001) == [1, 1]
        @test icdf(dist2, 1.0) == [3, 3]
        @test icdf(dist2, 0.01) == [1, 1]
        @test icdf(dist2, 0.99) == [3, 3]
        @test icdf(dist1, 0.5) == [2, 1]

        dist = Contingency([0.1 0.2; 0.5 0.2])
        @test icdf(dist, 0.1) == [1, 1]
        @test icdf(dist, 0.2) == [1, 2]
        @test icdf(dist, 0.4) == [2, 1]
        @test icdf(dist, 0.5) == [2, 1]
        @test icdf(dist, 0.7) == [2, 2]
    end

    @testset "rand" begin
        dist = Contingency([0.3 0.2; 0.1 0.4])
        nsamples = 1000
        rng = collect(1:100)
        for i in 1:100
            samples = rand(MersenneTwister(rng[i]), dist, nsamples)
            mestimated = mean(samples)
            @test isapprox(mestimated, mean(dist), atol = 1e-1)
            @test isapprox(
                sum((sample - mestimated) * (sample - mestimated)' for sample in samples) / (nsamples),
                cov(dist),
                atol = 1e-1
            )
        end
    end

    @testset "prod KnownExponentialFamilyDistribution" begin
        for i in 2:50
            p1 = rand(i, i)
            p2 = rand(i, i)
            p1 = p1 ./ sum(p1)
            p2 = p2 ./ sum(p2)
            distleft = Contingency(p1)
            distright = Contingency(p2)
            efleft = convert(KnownExponentialFamilyDistribution, distleft)
            efright = convert(KnownExponentialFamilyDistribution, distright)

            ηleft = getnaturalparameters(efleft)
            ηright = getnaturalparameters(efright)

            efprod = prod(efleft, efright)
            distprod = prod(ClosedProd(), distleft, distright)
            @test efprod == KnownExponentialFamilyDistribution(Contingency, ηleft + ηright)
            @test distprod ≈ convert(Distribution, efprod)
        end
    end
end

end
