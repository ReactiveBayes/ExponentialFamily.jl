module WishartTest

using Test
using ExponentialFamily
using Distributions
using Random
using LinearAlgebra
using PDMats

import ExponentialFamily: WishartMessage, NaturalParameters, get_params
import StatsFuns: logmvgamma

@testset "Wishart" begin

    # Wishart comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "mean(::logdet)" begin
        @test mean(logdet, Wishart(3, [1.0 0.0; 0.0 1.0])) ≈ 0.845568670196936
        @test mean(
            logdet,
            Wishart(
                5,
                [
                    1.4659658963311604 1.111775094889733 0.8741034114800605
                    1.111775094889733 0.8746971141492232 0.6545661366809246
                    0.8741034114800605 0.6545661366809246 0.5498917856395482
                ]
            )
        ) ≈ -3.4633310802040693
    end

    @testset "mean(::cholinv)" begin
        L    = rand(2, 2)
        S    = L * L' + diageye(2)
        invS = cholinv(S)
        @test mean(inv, Wishart(5, S)) ≈ mean(InverseWishart(5, invS))
        @test mean(cholinv, Wishart(5, S)) ≈ mean(InverseWishart(5, invS))
    end

    @testset "vague" begin
        @test_throws MethodError vague(Wishart)

        d = vague(Wishart, 3)

        @test typeof(d) <: Wishart
        @test mean(d) == Matrix(Diagonal(3 * 1e12 * ones(3)))
    end

    @testset "prod" begin
        inv_v1 = cholinv([9.0 -3.4; -3.4 11.0])
        inv_v2 = cholinv([10.2 -3.3; -3.3 5.0])
        inv_v3 = cholinv([8.1 -2.7; -2.7 9.0])

        @test prod(ProdAnalytical(), WishartMessage(3, inv_v1), WishartMessage(3, inv_v2)) ≈
              WishartMessage(
            3,
            cholinv([4.776325721474591 -1.6199382410125422; -1.6199382410125422 3.3487476649765537])
        )
        @test prod(ProdAnalytical(), WishartMessage(4, inv_v1), WishartMessage(4, inv_v3)) ≈
              WishartMessage(5, cholinv([4.261143738311623 -1.5064864332819319; -1.5064864332819319 4.949867121624725]))
        @test prod(ProdAnalytical(), WishartMessage(5, inv_v2), WishartMessage(4, inv_v3)) ≈
              WishartMessage(6, cholinv([4.51459128065395 -1.4750681198910067; -1.4750681198910067 3.129155313351499]))
    end

    @testset "ndims" begin
        @test ndims(vague(Wishart, 3)) === 3
        @test ndims(vague(Wishart, 4)) === 4
        @test ndims(vague(Wishart, 5)) === 5
    end

    @testset "WishartNaturalParameters" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(
                    Distribution,
                    NaturalParameters(Wishart, [3.0, [-i 0.0; 0.0 -i]])
                ) ≈
                      Wishart(9.0, -0.5 * inv([-i 0.0; 0.0 -i]))
                @test convert(
                    Distribution,
                    NaturalParameters(WishartMessage, [3.0, [-i 0.0; 0.0 -i]])
                ) ≈
                      WishartMessage(9.0, -0.5 * inv([-i 0.0; 0.0 -i]))
            end
        end

        @testset "logpdf" begin
            for i in 1:10
                wishart_np = NaturalParameters(Wishart, [3.0, [-i 0.0; 0.0 -i]])
                distribution = Wishart(9.0, -0.5 * inv([-i 0.0; 0.0 -i]))
                @test logpdf(distribution, [1.0 0.0; 0.0 1.0]) ≈ logpdf(wishart_np, [1.0 0.0; 0.0 1.0])
                @test logpdf(distribution, [1.0 0.2; 0.2 1.0]) ≈ logpdf(wishart_np, [1.0 0.2; 0.2 1.0])
                @test logpdf(distribution, [1.0 -0.1; -0.1 3.0]) ≈ logpdf(wishart_np, [1.0 -0.1; -0.1 3.0])
            end
        end

        @testset "lognormalizer" begin
            @test lognormalizer(NaturalParameters(Wishart, [3.0, [-1.0 0.0; 0.0 -1.0]])) ≈
                  logmvgamma(2, 3.0 + (2 + 1) / 2)
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(NaturalParameters(Wishart, [3.0, [-i 0.0; 0.0 -i]])) === true
                @test isproper(NaturalParameters(Wishart, [3.0, [i 0.0; 0.0 -i]])) === false
                @test isproper(NaturalParameters(Wishart, [-1.0, [-i 0.0; 0.0 -i]])) === false
            end
        end
    end
end

end
