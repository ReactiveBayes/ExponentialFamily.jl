module VonMisesFisherTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff

import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, logpartition, basemeasure, fisherinformation, norm
import SpecialFunctions: besseli
import StatsFuns: softmax
@testset "VonMisesFisher" begin

    # VonMisesFisher comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(VonMisesFisher, 3)

        @test typeof(d) <: VonMisesFisher
        @test mean(d) == zeros(3)
        @test params(d) == (zeros(3), 1.0e-12)
    end

    @testset "prod" begin
        @test prod(ClosedProd(), VonMisesFisher([sin(30), cos(30)], 3.0), VonMisesFisher([sin(45), cos(45)], 4.0)) ≈
              Base.convert(
            Distribution,
            prod(convert(ExponentialFamilyDistribution, VonMisesFisher([sin(30), cos(30)], 3.0)),
                convert(ExponentialFamilyDistribution, VonMisesFisher([sin(45), cos(45)], 4.0)))
        )
        @test prod(ClosedProd(), VonMisesFisher([sin(15), cos(15)], 5.0), VonMisesFisher([cos(20), sin(20)], 2.0)) ≈
              Base.convert(
            Distribution,
            prod(convert(ExponentialFamilyDistribution, VonMisesFisher([sin(15), cos(15)], 5.0)),
                convert(ExponentialFamilyDistribution, VonMisesFisher([cos(20), sin(20)], 2.0)))
        )
    end

    @testset "natural parameters related" begin
        @testset "Constructor" begin
            for i in 1:6:360
                @test convert(Distribution, ExponentialFamilyDistribution(VonMisesFisher, 3 * [cos(i), sin(i)])) ≈
                      VonMisesFisher([cos(i), sin(i)], 3)

                @test convert(ExponentialFamilyDistribution, VonMisesFisher([cos(i), sin(i)], 3)) ≈
                      ExponentialFamilyDistribution(VonMisesFisher, 3 * [cos(i), sin(i)])
            end
        end

        @testset "logpartition" begin
            @test logpartition(ExponentialFamilyDistribution(VonMisesFisher, 2 * [sin(15), cos(15)])) ≈
                  log(besseli(0, 2.0))
            @test logpartition(ExponentialFamilyDistribution(VonMisesFisher, 6 * [sin(25), cos(25)])) ≈
                  log(besseli(0, 6.0))
        end

        @testset "logpdf" begin
            for i in 1:10, j in 1:10
                @test logpdf(
                    ExponentialFamilyDistribution(VonMisesFisher, 3 * [cos(i), sin(i)]),
                    [0.01, sqrt(1 - 0.01^2)]
                ) ≈
                      logpdf(VonMisesFisher([cos(i), sin(i)], 3), [0.01, sqrt(1 - 0.01^2)])
                @test logpdf(
                    ExponentialFamilyDistribution(VonMisesFisher, 3 * [cos(2 * i), sin(2 * i)]),
                    [0.5, sqrt(1 - 0.5^2)]
                ) ≈
                      logpdf(VonMisesFisher([cos(2 * i), sin(2 * i)], 3), [0.5, sqrt(1 - 0.5^2)])
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(ExponentialFamilyDistribution(VonMisesFisher, [i, i])) === true
            end
            for i in 1:10
                @test isproper(ExponentialFamilyDistribution(VonMisesFisher, [-i, -i])) === false
            end
            @test isproper(ExponentialFamilyDistribution(VonMisesFisher, [0, 0])) === true
        end

        @testset "fisher information" begin
            function transformation(params)
                κ = sqrt(params' * params)
                μ = params / κ
                return [μ; κ]
            end

            for l in 2:10, κ in 0.001:2.0:30.0
                μ = rand(l)
                μ = μ / norm(μ)
                dist = VonMisesFisher(μ, κ)
                ef = convert(ExponentialFamilyDistribution, dist)
                η = getnaturalparameters(ef)

                f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(VonMisesFisher, η))
                autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
                @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-7
                J = ForwardDiff.jacobian(transformation, η)
                @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) atol = 1e-7
            end
        end
    end

    @testset "ExponentialFamilyDistribution mean" begin
        for l in 2:10, κ in 0.001:2.0:30.0
            μ = rand(l)
            μ = μ / norm(μ)
            dist = VonMisesFisher(μ, κ)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
        end
    end
end

end
