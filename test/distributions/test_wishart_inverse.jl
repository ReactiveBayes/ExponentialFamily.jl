module InverseWishartTest

using Test
using ExponentialFamily
using Distributions
using Random
using LinearAlgebra
using StableRNGs
using ForwardDiff

import ExponentialFamily:
    InverseWishartFast, ExponentialFamilyDistribution, getnaturalparameters, basemeasure,
    fisherinformation, logpartition, value_support, variate_form
import Distributions: pdf!
import StatsFuns: logmvgamma

include("../testutils.jl")

@testset "InverseWishartFast" begin
    @testset "common" begin
        @test InverseWishartFast <: Distribution
        @test InverseWishartFast <: ContinuousDistribution
        @test InverseWishartFast <: MatrixDistribution

        @test value_support(InverseWishartFast) === Continuous
        @test variate_form(InverseWishartFast) === Matrixvariate
    end

    @testset "ExponentialFamilyDistribution{InverseWishartFast}" begin
        @testset for dim in (3), S in rand(InverseWishart(10, Eye(dim)), 2)
            ν = dim + 4
            @testset let (d = InverseWishartFast(ν, S))
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false, test_fisherinformation_against_hessian = false)
                (η1, η2) = unpack_parameters(InverseWishartFast, getnaturalparameters(ef))

                for x in Eye(dim)
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === 1.0
                    @test @inferred(sufficientstatistics(ef, x)) === (logdet(x), inv(x))
                    @test @inferred(logpartition(ef)) ≈ (η1 + (dim + 1) / 2) * logdet(-η2) + logmvgamma(dim, -(η1 + (dim + 1) / 2))
                end
            end
        end
    end

    @testset "statistics" begin
        rng = StableRNG(42)
        # ν > dim(d) + 1
        for ν in 4:10
            L = randn(rng, ν - 2, ν - 2)
            S = L * L'
            d = InverseWishartFast(ν, S)

            @test mean(d) == mean(InverseWishart(params(d)...))
            @test mode(d) == mode(InverseWishart(params(d)...))
        end

        # ν > dim(d) + 3
        for ν in 5:10
            L = randn(rng, ν - 4, ν - 4)
            S = L * L'
            d = InverseWishartFast(ν, S)

            @test cov(d) == cov(InverseWishart(params(d)...))
            @test var(d) == var(InverseWishart(params(d)...))
        end
    end

    @testset "vague" begin
        @test_throws MethodError vague(InverseWishartFast)

        dims = 3
        d1 = vague(InverseWishart, dims)

        @test typeof(d1) <: InverseWishart
        ν1, S1 = params(d1)
        @test ν1 == dims + 2
        @test S1 == tiny .* Eye(dims)

        @test mean(d1) == S1

        dims = 4
        d2 = vague(InverseWishart, dims)

        @test typeof(d2) <: InverseWishart
        ν2, S2 = params(d2)
        @test ν2 == dims + 2
        @test S2 == tiny .* Eye(dims)

        @test mean(d2) == S2
    end

    @testset "entropy" begin
        @test entropy(
            InverseWishartFast(
                2.0,
                [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
            )
        ) ≈ 10.111427477184794
        @test entropy(InverseWishartFast(5.0, Eye(4))) ≈ 8.939145914882221
    end

    @testset "convert" begin
        rng = StableRNG(42)
        for ν in 2:10
            L = randn(rng, ν, ν)
            S = L * L'
            d = InverseWishartFast(ν, S)
            @test convert(InverseWishart, d) == InverseWishart(ν, S)
        end
    end

    @testset "mean(::typeof(logdet))" begin
        rng = StableRNG(123)
        ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(logdet, InverseWishartFast(ν, S)), mean(logdet.(samples)), atol = 1e-2)

        ν, S = 4.0, Eye(3)
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(logdet, InverseWishartFast(ν, S)), mean(logdet.(samples)), atol = 1e-2)
    end

    @testset "mean(::typeof(inv))" begin
        rng = StableRNG(321)
        ν, S = 2.0, [2.2658069783329573 -0.47934965873423374; -0.47934965873423374 1.4313564100863712]
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(inv, InverseWishartFast(ν, S)), mean(inv.(samples)), atol = 1e-2)

        ν, S = 4.0, Eye(3)
        samples = rand(rng, InverseWishart(ν, S), Int(1e6))
        @test isapprox(mean(inv, InverseWishartFast(ν, S)), mean(inv.(samples)), atol = 1e-2)
    end

    @testset "prod" begin
        d1 = InverseWishartFast(3.0, Eye(2))
        d2 = InverseWishartFast(-3.0, [0.6423504672769315 0.9203141654948761; 0.9203141654948761 1.528137747462735])

        @test prod(PreserveTypeProd(Distribution), d1, d2) ≈
              InverseWishartFast(3.0, [1.6423504672769313 0.9203141654948761; 0.9203141654948761 2.528137747462735])

        d1 = InverseWishartFast(4.0, Eye(3))
        d2 = InverseWishartFast(-2.0, Eye(3))

        @test prod(PreserveTypeProd(Distribution), d1, d2) ≈ InverseWishartFast(6.0, 2 * Eye(3))
    end

    @testset "rand" begin
        for d in (2, 3, 4, 5)
            v = rand() + d
            L = rand(d, d)
            S = L' * L + d * Eye(d)
            cS = copy(S)
            container1 = [zeros(d, d) for _ in 1:100]
            container2 = [zeros(d, d) for _ in 1:100]

            # Check in-place version
            @test rand!(StableRNG(321), InverseWishart(v, S), container1) ≈
                  rand!(StableRNG(321), InverseWishartFast(v, S), container2)

            # Check that the matrix has not been corrupted
            @test all(S .=== cS)

            # Check non-inplace version
            @test rand(StableRNG(321), InverseWishart(v, S), length(container1)) ≈
                  rand(StableRNG(321), InverseWishartFast(v, S), length(container2))
        end
    end

    @testset "pdf!" begin
        for d in (2, 3, 4, 5), n in (10, 20)
            v = rand() + d
            L = rand(d, d)
            S = L' * L + d * Eye(d)

            samples = map(1:n) do _
                L_sample = rand(d, d)
                return L_sample' * L_sample + d * Eye(d)
            end

            result = zeros(n)

            @test all(pdf(InverseWishart(v, S), samples) .≈ pdf!(result, InverseWishartFast(v, S), samples))
        end
    end

    @testset "prod with ExponentialFamilyDistribution{InverseWishartFast}" begin
        for Sleft in rand(InverseWishart(10, Eye(2)), 2), Sright in rand(InverseWishart(10, Eye(2)), 2), νright in (6, 7), νleft in (4, 5)
            let left = InverseWishartFast(νleft, Sleft), right = InverseWishartFast(νright, Sright)
                @test test_generic_simple_exponentialfamily_product(
                    left,
                    right,
                    strategies = (PreserveTypeProd(ExponentialFamilyDistribution{InverseWishartFast}), GenericProd())
                )
            end
        end
    end
end

end
