module ErlangTest

using Test
using ExponentialFamily
using Random
using Distributions

import SpecialFunctions: logfactorial
import ExponentialFamily: xtlog

@testset "Erlang" begin
    @testset "Constructor" begin

        @test Erlang() == Erlang(1, 1.0)
        @test Erlang(1.0, 2.0) == Erlang(1, 2.0)
        @test Erlang(1, 2) == Erlang(1, 2.0)
        @test Erlang(1.0, 2) == Erlang(1.0, 2.0)
        @test Erlang(1.0f0, 2.0f0) == Erlang(1, 2.0f0)
        @test Erlang(1.0f0, 2) == Erlang{Float32}(1, 2.0f0)
        @test Erlang(1.0f0, 2.0) == Erlang{Float64}(1, 2.0)

    end

    @testset "vague" begin
        vague(Erlang) == Erlang(1.0, 1e12)
    end

    @testset "Stats methods for Erlang" begin
        dist1 = Erlang(1, 1.0)

        @test mean(dist1) === 1.0
        @test var(dist1) === 1.0
        @test cov(dist1) === 1.0
        @test shape(dist1) === 1
        @test scale(dist1) === 1.0
        @test rate(dist1) === 1.0
        @test entropy(dist1) ≈ 1.0

        dist2 = Erlang(1, 2.0)

        @test mean(dist2) === 2.0
        @test var(dist2) === 4.0
        @test cov(dist2) === 4.0
        @test shape(dist2) === 1
        @test scale(dist2) === 2.0
        @test rate(dist2) === inv(2.0)
        @test entropy(dist2) ≈ 1.6931471805599454

        dist3 = Erlang(2, 2.0)

        @test mean(dist3) === 4.0
        @test var(dist3) === 8.0
        @test cov(dist3) === 8.0
        @test shape(dist3) === 2
        @test scale(dist3) === 2.0
        @test rate(dist3) === inv(2.0)
        @test entropy(dist3) ≈ 2.2703628454614764
    end

    @testset "ErlangNaturalParameters" begin
        for i in 2:10
            @test convert(Distribution, ErlangNaturalParameters(i, -i)) ≈ Erlang(i + 1, inv(i))
            @test Distributions.logpdf(ErlangNaturalParameters(i, -i), 10) ≈ Distributions.logpdf(Erlang(i + 1, inv(i)), 10)
            @test isproper(ErlangNaturalParameters(i, -i)) === true
            @test isproper(ErlangNaturalParameters(-i, i)) === false

            @test convert(ErlangNaturalParameters, i, -i) == ErlangNaturalParameters(i, -i)
            
            @test as_naturalparams(ErlangNaturalParameters, i, -i) == ErlangNaturalParameters(i, -i)
        end
    end

    @testset "Base methods" begin
        check_basic_statistics = (left, right) -> begin
            @test mean(left) ≈ mean(right)
            @test var(left) ≈ var(right)
            @test cov(left) ≈ cov(right)
            @test shape(left) ≈ shape(right)
            @test scale(left) ≈ scale(right)
            @test rate(left) ≈ rate(right)
            @test entropy(left) ≈ entropy(right)
            @test pdf(left, 1.0) ≈ pdf(right, 1.0)
            @test pdf(left, 10.0) ≈ pdf(right, 10.0)
            @test logpdf(left, 1.0) ≈ logpdf(right, 1.0)
            @test logpdf(left, 10.0) ≈ logpdf(right, 10.0)
            @test mean(log, left) ≈ mean(log, right)
        end

        types = ExponentialFamily.union_types(GammaDistributionsFamily{Float64})
        rng   = MersenneTwister(1234)

        for type in types
            left = convert(type, rand(rng, Float64), rand(rng, Float64))
            for type in types
                right = convert(type, left)
                check_basic_statistics(left, right)
            end
        end
    end

    @testset "prod" begin
        @test prod(ProdAnalytical(), Erlang(1, 1), Erlang(1, 1)) == Erlang(1, 1 / 2)
        @test prod(ProdAnalytical(), Erlang(1, 2), Erlang(1, 1)) == Erlang(1, 2 / 3)
        @test prod(ProdAnalytical(), Erlang(1, 2), Erlang(1, 2)) == Erlang(1, 1)
        @test prod(ProdAnalytical(), Erlang(2, 2), Erlang(1, 2)) == Erlang(2, 1)
        @test prod(ProdAnalytical(), Erlang(2, 2), Erlang(2, 2)) == Erlang(3, 1)
    end

    # TODO: lognormalizer etc
end

end
