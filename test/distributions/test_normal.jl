module NormalTest

using Test
using ExponentialFamily
using LinearAlgebra
using Distributions
using ForwardDiff
using Random
using StableRNGs

import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

@testset "Normal" begin
    @testset "Univariate conversions" begin
        check_basic_statistics =
            (left, right; include_extended_methods = true) -> begin
                @test mean(left) ≈ mean(right)
                @test median(left) ≈ median(right)
                @test mode(left) ≈ mode(right)
                @test var(left) ≈ var(right)
                @test std(left) ≈ std(right)
                @test entropy(left) ≈ entropy(right)

                for value in (1.0, -1.0, 0.0, mean(left), mean(right), rand())
                    @test pdf(left, value) ≈ pdf(right, value)
                    @test logpdf(left, value) ≈ logpdf(right, value)
                    @test all(
                        ForwardDiff.gradient((x) -> logpdf(left, x[1]), [value]) .≈
                        ForwardDiff.gradient((x) -> logpdf(right, x[1]), [value])
                    )
                    @test all(
                        ForwardDiff.hessian((x) -> logpdf(left, x[1]), [value]) .≈
                        ForwardDiff.hessian((x) -> logpdf(right, x[1]), [value])
                    )
                end

                # These methods are not defined for distributions from `Distributions.jl
                if include_extended_methods
                    @test cov(left) ≈ cov(right)
                    @test invcov(left) ≈ invcov(right)
                    @test weightedmean(left) ≈ weightedmean(right)
                    @test precision(left) ≈ precision(right)
                    @test all(mean_cov(left) .≈ mean_cov(right))
                    @test all(mean_invcov(left) .≈ mean_invcov(right))
                    @test all(mean_precision(left) .≈ mean_precision(right))
                    @test all(weightedmean_cov(left) .≈ weightedmean_cov(right))
                    @test all(weightedmean_invcov(left) .≈ weightedmean_invcov(right))
                    @test all(weightedmean_precision(left) .≈ weightedmean_precision(right))
                end
            end

        types  = ExponentialFamily.union_types(UnivariateNormalDistributionsFamily{Float64})
        etypes = ExponentialFamily.union_types(UnivariateNormalDistributionsFamily)

        rng = MersenneTwister(1234)

        for type in types
            left = convert(type, rand(rng, Float64), rand(rng, Float64))
            check_basic_statistics(left, convert(Normal, left); include_extended_methods = false)
            for type in [types..., etypes...]
                right = convert(type, left)
                check_basic_statistics(left, right)

                p1 = prod(ProdPreserveTypeLeft(), left, right)
                @test typeof(p1) <: typeof(left)

                p2 = prod(ProdPreserveTypeRight(), left, right)
                @test typeof(p2) <: typeof(right)

                p3 = prod(ClosedProd(), left, right)

                check_basic_statistics(p1, p2)
                check_basic_statistics(p2, p3)
                check_basic_statistics(p1, p3)
            end
        end
    end

    @testset "Multivariate conversions" begin
        check_basic_statistics =
            (left, right, dims; include_extended_methods = true) -> begin
                @test mean(left) ≈ mean(right)
                @test mode(left) ≈ mode(right)
                @test var(left) ≈ var(right)
                @test cov(left) ≈ cov(right)
                @test logdetcov(left) ≈ logdetcov(right)
                @test length(left) === length(right)
                @test size(left) === size(right)
                @test entropy(left) ≈ entropy(right)

                for value in (
                    fill(1.0, dims),
                    fill(-1.0, dims),
                    fill(0.1, dims),
                    mean(left),
                    mean(right),
                    rand(dims)
                )
                    @test pdf(left, value) ≈ pdf(right, value)
                    @test logpdf(left, value) ≈ logpdf(right, value)
                    @test all(
                        isapprox.(
                            ForwardDiff.gradient((x) -> logpdf(left, x), value),
                            ForwardDiff.gradient((x) -> logpdf(right, x), value),
                            atol = 1e-14
                        )
                    )
                    # TODO: test fails
                    @test all(
                        isapprox.(
                            ForwardDiff.hessian((x) -> logpdf(left, x), value),
                            ForwardDiff.hessian((x) -> logpdf(right, x), value),
                            atol = 1e-14
                        )
                    )
                end

                # These methods are not defined for distributions from `Distributions.jl
                if include_extended_methods
                    @test ndims(left) === ndims(right)
                    @test invcov(left) ≈ invcov(right)
                    @test weightedmean(left) ≈ weightedmean(right)
                    @test precision(left) ≈ precision(right)
                    @test all(mean_cov(left) .≈ mean_cov(right))
                    @test all(mean_invcov(left) .≈ mean_invcov(right))
                    @test all(mean_precision(left) .≈ mean_precision(right))
                    @test all(weightedmean_cov(left) .≈ weightedmean_cov(right))
                    @test all(weightedmean_invcov(left) .≈ weightedmean_invcov(right))
                    @test all(weightedmean_precision(left) .≈ weightedmean_precision(right))
                end
            end

        types  = ExponentialFamily.union_types(MultivariateNormalDistributionsFamily{Float64})
        etypes = ExponentialFamily.union_types(MultivariateNormalDistributionsFamily)

        dims = (2, 3, 5)
        rng  = MersenneTwister(1234)

        for dim in dims
            for type in types
                left = convert(type, rand(rng, Float64, dim), Matrix(Diagonal(abs.(rand(rng, Float64, dim)))))
                check_basic_statistics(left, convert(MvNormal, left), dim; include_extended_methods = false)
                for type in [types..., etypes...]
                    right = convert(type, left)
                    check_basic_statistics(left, right, dim)

                    p1 = prod(ProdPreserveTypeLeft(), left, right)
                    @test typeof(p1) <: typeof(left)

                    p2 = prod(ProdPreserveTypeRight(), left, right)
                    @test typeof(p2) <: typeof(right)

                    p3 = prod(ClosedProd(), left, right)

                    check_basic_statistics(p1, p2, dim)
                    check_basic_statistics(p2, p3, dim)
                    check_basic_statistics(p1, p3, dim)
                end
            end
        end
    end

    @testset "Variate forms promotions" begin
        @test promote_variate_type(Univariate, NormalMeanVariance) === NormalMeanVariance
        @test promote_variate_type(Univariate, NormalMeanPrecision) === NormalMeanPrecision
        @test promote_variate_type(Univariate, NormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

        @test promote_variate_type(Multivariate, NormalMeanVariance) === MvNormalMeanCovariance
        @test promote_variate_type(Multivariate, NormalMeanPrecision) === MvNormalMeanPrecision
        @test promote_variate_type(Multivariate, NormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision

        @test promote_variate_type(Univariate, MvNormalMeanCovariance) === NormalMeanVariance
        @test promote_variate_type(Univariate, MvNormalMeanPrecision) === NormalMeanPrecision
        @test promote_variate_type(Univariate, MvNormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

        @test promote_variate_type(Multivariate, MvNormalMeanCovariance) === MvNormalMeanCovariance
        @test promote_variate_type(Multivariate, MvNormalMeanPrecision) === MvNormalMeanPrecision
        @test promote_variate_type(Multivariate, MvNormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision
    end

    @testset "Sampling univariate" begin
        rng = MersenneTwister(1234)

        for T in (Float32, Float64)
            let # NormalMeanVariance
                μ, v = 10randn(rng), 10rand(rng)
                d    = convert(NormalMeanVariance{T}, μ, v)

                @test typeof(rand(d)) <: T

                samples = rand(rng, d, 5_000)

                @test isapprox(mean(samples), μ, atol = 0.5)
                @test isapprox(var(samples), v, atol = 0.5)
            end

            let # NormalMeanPrecision
                μ, w = 10randn(rng), 10rand(rng)
                d    = convert(NormalMeanPrecision{T}, μ, w)

                @test typeof(rand(d)) <: T

                samples = rand(rng, d, 5_000)

                @test isapprox(mean(samples), μ, atol = 0.5)
                @test isapprox(inv(var(samples)), w, atol = 0.5)
            end

            let # WeightedMeanPrecision
                wμ, w = 10randn(rng), 10rand(rng)
                d     = convert(NormalWeightedMeanPrecision{T}, wμ, w)

                @test typeof(rand(d)) <: T

                samples = rand(rng, d, 5_000)

                @test isapprox(inv(var(samples)) * mean(samples), wμ, atol = 0.5)
                @test isapprox(inv(var(samples)), w, atol = 0.5)
            end
        end
    end

    @testset "Sampling multivariate" begin
        rng = MersenneTwister(1234)
        for n in (2, 3), T in (Float64,), nsamples in (10_000,)
            μ = randn(rng, n)
            L = randn(rng, n, n)
            Σ = L * L'

            d = convert(MvNormalMeanCovariance{T}, μ, Σ)
            @test typeof(rand(d)) <: Vector{T}

            samples = eachcol(rand(rng, d, nsamples))
            weights = fill(1 / nsamples, nsamples)

            @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
            @test isapprox(
                sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
                cov(d),
                atol = n * 0.5
            )

            μ = randn(rng, n)
            L = randn(rng, n, n)
            W = L * L'
            d = convert(MvNormalMeanCovariance{T}, μ, W)
            @test typeof(rand(d)) <: Vector{T}

            samples = eachcol(rand(rng, d, nsamples))
            weights = fill(1 / nsamples, nsamples)

            @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
            @test isapprox(
                sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
                cov(d),
                atol = n * 0.5
            )

            ξ = randn(rng, n)
            L = randn(rng, n, n)
            W = L * L'

            d = convert(MvNormalWeightedMeanPrecision{T}, ξ, W)

            @test typeof(rand(d)) <: Vector{T}

            samples = eachcol(rand(rng, d, nsamples))
            weights = fill(1 / nsamples, nsamples)

            @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
            @test isapprox(
                sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
                cov(d),
                atol = n * 0.5
            )
        end
    end

    @testset "UnivariateNormalKnownExponentialFamilyDistribution" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(Distribution, KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [i, -i])) ==
                      NormalWeightedMeanPrecision(i, 2 * i)
                @test convert(KnownExponentialFamilyDistribution, NormalWeightedMeanPrecision(i, 2 * i)) ≈
                      KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, float([i, -i]))
                @test convert(KnownExponentialFamilyDistribution, NormalWeightedMeanPrecision(i, 2 * i)) ≈
                      KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, float([i, -i]))
            end
        end

        @testset "logpartition" begin
            @test logpartition(KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [1, -2])) ≈
                  -(log(2) - 1 / 8)
        end

        @testset "logpdf" begin
            for i in 1:10
                @test logpdf(KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [i, -i]), 0) ≈
                      logpdf(NormalWeightedMeanPrecision(i, 2 * i), 0)
            end
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [i, -i])) === true
                @test isproper(KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [i, i])) === false
            end
        end

        @testset "fisherinformation" begin
            for (η1, η2) in Iterators.product(1:10, -10:1:-1)
                ef = KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [η1, η2])
                f_logpartion = (η) -> logpartition(KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, η))
                autograd_inforamation_matrix = (η) -> ForwardDiff.hessian(f_logpartion, η)
                @test fisherinformation(ef) ≈ autograd_inforamation_matrix([η1, η2])
            end
        end
    end

    @testset "MultivariateNormalKnownExponentialFamilyDistribution" begin
        @testset "Constructor" begin
            for i in 1:10
                @test convert(
                    Distribution,
                    KnownExponentialFamilyDistribution(MvGaussianWeightedMeanPrecision, [[i, 0], [-i 0; 0 -i]])
                ) ==
                      MvGaussianWeightedMeanPrecision([i, 0], [2*i 0; 0 2*i])

                @test convert(
                    KnownExponentialFamilyDistribution,
                    MvGaussianWeightedMeanPrecision([i, 0], [2*i 0; 0 2*i])
                ) ≈
                      KnownExponentialFamilyDistribution(
                    MvGaussianWeightedMeanPrecision,
                    [float([i, 0]), float([-i 0; 0 -i])]
                )
            end
        end

        @testset "logpdf" begin
            for i in 1:10
                mv_np = KnownExponentialFamilyDistribution(MvGaussianWeightedMeanPrecision, [[i, 0], [-i 0; 0 -i]])
                distribution = MvGaussianWeightedMeanPrecision([i, 0.0], [2*i -0.0; -0.0 2*i])
                @test logpdf(distribution, [0.0, 0.0]) ≈ logpdf(mv_np, [0.0, 0.0])
                @test logpdf(distribution, [1.0, 0.0]) ≈ logpdf(mv_np, [1.0, 0.0])
                @test logpdf(distribution, [1.0, 1.0]) ≈ logpdf(mv_np, [1.0, 1.0])
            end
        end

        @testset "logpartition" begin
            @test logpartition(KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [1, -2])) ≈
                  -(log(2) - 1 / 8)
            dist = MvNormalMeanPrecision(ones(2), diageye(2))
            ef = convert(KnownExponentialFamilyDistribution, dist)
            logpartition(ef) = 1.0
        end

        @testset "isproper" begin
            for i in 1:10
                @test isproper(KnownExponentialFamilyDistribution(MvNormalMeanCovariance, [[i, 0], [-i 0; 0 -i]])) ===
                      true
                @test isproper(KnownExponentialFamilyDistribution(MvNormalMeanCovariance, [[i, 0], [i 0; 0 i]])) ===
                      false
            end
        end

        @testset "basemeasure" begin
            for i in 1:10
                @test basemeasure(
                    KnownExponentialFamilyDistribution(MvNormalMeanCovariance, [[i, 0], [-i 0; 0 -i]]),
                    rand(2)
                ) ==
                      1 / (2pi)
            end
        end
    end
end

end
