module NormalTest

using ExponentialFamily, Distributions
using Test, LinearAlgebra, ForwardDiff, Random, StableRNGs, SparseArrays

include("../../testutils.jl")

import ExponentialFamily: promote_variate_type

@testset "Normal family" begin

    # @testset "Univariate conversions" begin
    #     check_basic_statistics =
    #         (left, right; include_extended_methods = true) -> begin
    #             @test mean(left) ≈ mean(right)
    #             @test median(left) ≈ median(right)
    #             @test mode(left) ≈ mode(right)
    #             @test var(left) ≈ var(right)
    #             @test std(left) ≈ std(right)
    #             @test entropy(left) ≈ entropy(right)

    #             for value in (1.0, -1.0, 0.0, mean(left), mean(right), rand())
    #                 @test pdf(left, value) ≈ pdf(right, value)
    #                 @test logpdf(left, value) ≈ logpdf(right, value)
    #                 @test all(
    #                     ForwardDiff.gradient((x) -> logpdf(left, x[1]), [value]) .≈
    #                     ForwardDiff.gradient((x) -> logpdf(right, x[1]), [value])
    #                 )
    #                 @test all(
    #                     ForwardDiff.hessian((x) -> logpdf(left, x[1]), [value]) .≈
    #                     ForwardDiff.hessian((x) -> logpdf(right, x[1]), [value])
    #                 )
    #             end

    #             # These methods are not defined for distributions from `Distributions.jl
    #             if include_extended_methods
    #                 @test cov(left) ≈ cov(right)
    #                 @test invcov(left) ≈ invcov(right)
    #                 @test weightedmean(left) ≈ weightedmean(right)
    #                 @test precision(left) ≈ precision(right)
    #                 @test all(mean_cov(left) .≈ mean_cov(right))
    #                 @test all(mean_invcov(left) .≈ mean_invcov(right))
    #                 @test all(mean_precision(left) .≈ mean_precision(right))
    #                 @test all(weightedmean_cov(left) .≈ weightedmean_cov(right))
    #                 @test all(weightedmean_invcov(left) .≈ weightedmean_invcov(right))
    #                 @test all(weightedmean_precision(left) .≈ weightedmean_precision(right))
    #             end
    #         end

    #     types  = ExponentialFamily.union_types(UnivariateNormalDistributionsFamily{Float64})
    #     etypes = ExponentialFamily.union_types(UnivariateNormalDistributionsFamily)

    #     rng = MersenneTwister(1234)

    #     for type in types
    #         left = convert(type, rand(rng, Float64), rand(rng, Float64))
    #         check_basic_statistics(left, convert(Normal, left); include_extended_methods = false)
    #         for type in [types..., etypes...]
    #             right = convert(type, left)
    #             check_basic_statistics(left, right)

    #             p1 = prod(PreserveTypeLeftProd(), left, right)
    #             @test typeof(p1) <: typeof(left)

    #             p2 = prod(PreserveTypeRightProd(), left, right)
    #             @test typeof(p2) <: typeof(right)

    #             p3 = prod(ClosedProd(), left, right)

    #             check_basic_statistics(p1, p2)
    #             check_basic_statistics(p2, p3)
    #             check_basic_statistics(p1, p3)
    #         end
    #     end
    # end

    # @testset "Multivariate conversions" begin
    #     check_basic_statistics =
    #         (left, right, dims; include_extended_methods = true) -> begin
    #             @test mean(left) ≈ mean(right)
    #             @test mode(left) ≈ mode(right)
    #             @test var(left) ≈ var(right)
    #             @test cov(left) ≈ cov(right)
    #             @test logdetcov(left) ≈ logdetcov(right)
    #             @test length(left) === length(right)
    #             @test size(left) === size(right)
    #             @test entropy(left) ≈ entropy(right)

    #             for value in (
    #                 fill(1.0, dims),
    #                 fill(-1.0, dims),
    #                 fill(0.1, dims),
    #                 mean(left),
    #                 mean(right),
    #                 rand(dims)
    #             )
    #                 @test pdf(left, value) ≈ pdf(right, value)
    #                 @test logpdf(left, value) ≈ logpdf(right, value)
    #                 @test all(
    #                     isapprox.(
    #                         ForwardDiff.gradient((x) -> logpdf(left, x), value),
    #                         ForwardDiff.gradient((x) -> logpdf(right, x), value),
    #                         atol = 1e-14
    #                     )
    #                 )
    #                 @test all(
    #                     isapprox.(
    #                         ForwardDiff.hessian((x) -> logpdf(left, x), value),
    #                         ForwardDiff.hessian((x) -> logpdf(right, x), value),
    #                         atol = 1e-14
    #                     )
    #                 )
    #             end

    #             # These methods are not defined for distributions from `Distributions.jl
    #             if include_extended_methods
    #                 @test ndims(left) === ndims(right)
    #                 @test invcov(left) ≈ invcov(right)
    #                 @test weightedmean(left) ≈ weightedmean(right)
    #                 @test precision(left) ≈ precision(right)
    #                 @test all(mean_cov(left) .≈ mean_cov(right))
    #                 @test all(mean_invcov(left) .≈ mean_invcov(right))
    #                 @test all(mean_precision(left) .≈ mean_precision(right))
    #                 @test all(weightedmean_cov(left) .≈ weightedmean_cov(right))
    #                 @test all(weightedmean_invcov(left) .≈ weightedmean_invcov(right))
    #                 @test all(weightedmean_precision(left) .≈ weightedmean_precision(right))
    #             end
    #         end

    #     types  = ExponentialFamily.union_types(MultivariateNormalDistributionsFamily{Float64})
    #     etypes = ExponentialFamily.union_types(MultivariateNormalDistributionsFamily)

    #     dims = (2, 3, 5)
    #     rng  = MersenneTwister(1234)

    #     for dim in dims
    #         for type in types
    #             left = convert(type, rand(rng, Float64, dim), Matrix(Diagonal(abs.(rand(rng, Float64, dim)))))
    #             check_basic_statistics(left, convert(MvNormal, left), dim; include_extended_methods = false)
    #             for type in [types..., etypes...]
    #                 right = convert(type, left)
    #                 check_basic_statistics(left, right, dim)

    #                 p1 = prod(PreserveTypeLeftProd(), left, right)
    #                 @test typeof(p1) <: typeof(left)

    #                 p2 = prod(PreserveTypeRightProd(), left, right)
    #                 @test typeof(p2) <: typeof(right)

    #                 p3 = prod(ClosedProd(), left, right)

    #                 check_basic_statistics(p1, p2, dim)
    #                 check_basic_statistics(p2, p3, dim)
    #                 check_basic_statistics(p1, p3, dim)
    #             end
    #         end
    #     end
    # end

    # @testset "Variate forms promotions" begin
    #     @test promote_variate_type(Univariate, NormalMeanVariance) === NormalMeanVariance
    #     @test promote_variate_type(Univariate, NormalMeanPrecision) === NormalMeanPrecision
    #     @test promote_variate_type(Univariate, NormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

    #     @test promote_variate_type(Multivariate, NormalMeanVariance) === MvNormalMeanCovariance
    #     @test promote_variate_type(Multivariate, NormalMeanPrecision) === MvNormalMeanPrecision
    #     @test promote_variate_type(Multivariate, NormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision

    #     @test promote_variate_type(Univariate, MvNormalMeanCovariance) === NormalMeanVariance
    #     @test promote_variate_type(Univariate, MvNormalMeanPrecision) === NormalMeanPrecision
    #     @test promote_variate_type(Univariate, MvNormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

    #     @test promote_variate_type(Multivariate, MvNormalMeanCovariance) === MvNormalMeanCovariance
    #     @test promote_variate_type(Multivariate, MvNormalMeanPrecision) === MvNormalMeanPrecision
    #     @test promote_variate_type(Multivariate, MvNormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision
    # end

    # @testset "Sampling univariate" begin
    #     rng = MersenneTwister(1234)

    #     for T in (Float32, Float64)
    #         let # NormalMeanVariance
    #             μ, v = 10randn(rng), 10rand(rng)
    #             d    = convert(NormalMeanVariance{T}, μ, v)

    #             @test typeof(rand(d)) <: T

    #             samples = rand(rng, d, 5_000)

    #             @test isapprox(mean(samples), μ, atol = 0.5)
    #             @test isapprox(var(samples), v, atol = 0.5)
    #         end

    #         let # NormalMeanPrecision
    #             μ, w = 10randn(rng), 10rand(rng)
    #             d    = convert(NormalMeanPrecision{T}, μ, w)

    #             @test typeof(rand(d)) <: T

    #             samples = rand(rng, d, 5_000)

    #             @test isapprox(mean(samples), μ, atol = 0.5)
    #             @test isapprox(inv(var(samples)), w, atol = 0.5)
    #         end

    #         let # WeightedMeanPrecision
    #             wμ, w = 10randn(rng), 10rand(rng)
    #             d     = convert(NormalWeightedMeanPrecision{T}, wμ, w)

    #             @test typeof(rand(d)) <: T

    #             samples = rand(rng, d, 5_000)

    #             @test isapprox(inv(var(samples)) * mean(samples), wμ, atol = 0.5)
    #             @test isapprox(inv(var(samples)), w, atol = 0.5)
    #         end
    #     end
    # end

    # @testset "Sampling multivariate" begin
    #     rng = MersenneTwister(1234)
    #     for n in (2, 3), T in (Float64,), nsamples in (10_000,)
    #         μ = randn(rng, n)
    #         L = randn(rng, n, n)
    #         Σ = L * L'

    #         d = convert(MvNormalMeanCovariance{T}, μ, Σ)
    #         @test typeof(rand(d)) <: Vector{T}

    #         samples = eachcol(rand(rng, d, nsamples))
    #         weights = fill(1 / nsamples, nsamples)

    #         @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
    #         @test isapprox(
    #             sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
    #             cov(d),
    #             atol = n * 0.5
    #         )

    #         μ = randn(rng, n)
    #         L = randn(rng, n, n)
    #         W = L * L'
    #         d = convert(MvNormalMeanCovariance{T}, μ, W)
    #         @test typeof(rand(d)) <: Vector{T}

    #         samples = eachcol(rand(rng, d, nsamples))
    #         weights = fill(1 / nsamples, nsamples)

    #         @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
    #         @test isapprox(
    #             sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
    #             cov(d),
    #             atol = n * 0.5
    #         )

    #         ξ = randn(rng, n)
    #         L = randn(rng, n, n)
    #         W = L * L'

    #         d = convert(MvNormalWeightedMeanPrecision{T}, ξ, W)

    #         @test typeof(rand(d)) <: Vector{T}

    #         samples = eachcol(rand(rng, d, nsamples))
    #         weights = fill(1 / nsamples, nsamples)

    #         @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
    #         @test isapprox(
    #             sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
    #             cov(d),
    #             atol = n * 0.5
    #         )
    #     end
    # end

    # @testset "ExponentialFamilyDistribution{NormalMeanVariance}" begin
    #     @testset for μ in -10.0:5.0:10.0, σ² in 0.1:1.0:5.0, T in ExponentialFamily.union_types(UnivariateNormalDistributionsFamily)
    #         @testset let d = convert(T, NormalMeanVariance(μ, σ²))

    #             ef = test_exponentialfamily_interface(d)

    #             (η₁, η₂) = (mean(d) / var(d), -1/2var(d))

    #             for x in 10randn(4)
    #                 @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
    #                 @test @inferred(basemeasure(ef, x)) ≈ 1 / sqrt(2π)
    #                 @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x, abs2(x)))
    #                 @test @inferred(logpartition(ef)) ≈ (-η₁^2/4η₂ - 1/2*log(-2η₂))
    #                 @test @inferred(insupport(ef, x))
    #             end

    #         end
    #     end

    #     # Test failing isproper cases
    #     @test !isproper(MeanParametersSpace(), NormalMeanVariance, [-1])
    #     @test !isproper(MeanParametersSpace(), NormalMeanVariance, [1, -0.1])
    #     @test !isproper(MeanParametersSpace(), NormalMeanVariance, [-0.1, -1])
    #     @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [-1.1])
    #     @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [1, 1])
    #     @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [-1.1, 1])

    # end

    # @testset "prod with ExponentialFamilyDistribution{NormalMeanVariance}" for μleft in 10randn(4), σ²left in 10rand(4), μright in 10randn(4), σ²right in 10rand(4), Tleft in ExponentialFamily.union_types(UnivariateNormalDistributionsFamily), Tright in ExponentialFamily.union_types(UnivariateNormalDistributionsFamily)
    #     @testset let (left, right) = (convert(Tleft, NormalMeanVariance(μleft, σ²left)), convert(Tright, NormalMeanVariance(μright, σ²right)))
    #         @test test_generic_simple_exponentialfamily_product(
    #             left,
    #             right,
    #             strategies = (
    #                 ClosedProd(),
    #                 GenericProd(),
    #                 PreserveTypeProd(ExponentialFamilyDistribution),
    #                 PreserveTypeProd(ExponentialFamilyDistribution{NormalMeanVariance})
    #             )
    #         )
    #     end
    # end

    @testset "ExponentialFamilyDistribution{MvNormalMeanCovariance}" begin
        @testset for s in (2, 3, ), T in ExponentialFamily.union_types(MultivariateNormalDistributionsFamily)
            μ = 10randn(s)
            L = randn(s, s)
            Σ = L * L'
            @testset let d = convert(T, MvNormalMeanCovariance(μ, Σ))

                ef = test_exponentialfamily_interface(
                    d;
                    test_distribution_conversion = false,
                    test_basic_functions = false,
                    test_fisherinformation_against_hessian = false,
                    test_fisherinformation_against_jacobian = false
                )

                run_test_distribution_conversion(d; assume_no_allocations = false)
                run_test_basic_functions(d; assume_no_allocations = false)
                # run_test_fisherinformation_against_hessian(d; assume_no_allocations = false)
                # run_test_fisherinformation_against_jacobian(d; assume_no_allocations = false)

                # (η₁, η₂) = (mean(d) / var(d), -1/2var(d))

                # for x in [ 10randn(s) in 1:4 ]
                #     @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                #     @test @inferred(basemeasure(ef, x)) ≈ 1 / sqrt(2π)
                #     @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x, abs2(x)))
                #     @test @inferred(logpartition(ef)) ≈ (-η₁^2/4η₂ - 1/2*log(-2η₂))
                #     @test @inferred(insupport(ef, x))
                # end

            end
        end

        # Test failing isproper cases
        # @test !isproper(MeanParametersSpace(), NormalMeanVariance, [-1])
        # @test !isproper(MeanParametersSpace(), NormalMeanVariance, [1, -0.1])
        # @test !isproper(MeanParametersSpace(), NormalMeanVariance, [-0.1, -1])
        # @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [-1.1])
        # @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [1, 1])
        # @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [-1.1, 1])

    end

    # @testset "univariate natural parameters related" begin
    #     @testset "Constructor" begin
    #         for i in 1:10
    #             @test convert(Distribution, ExponentialFamilyDistribution(NormalWeightedMeanPrecision, [i, -i])) ==
    #                   NormalWeightedMeanPrecision(i, 2 * i)
    #             @test convert(ExponentialFamilyDistribution, NormalWeightedMeanPrecision(i, 2 * i)) ≈
    #                   ExponentialFamilyDistribution(NormalWeightedMeanPrecision, float([i, -i]))
    #             @test convert(ExponentialFamilyDistribution, NormalWeightedMeanPrecision(i, 2 * i)) ≈
    #                   ExponentialFamilyDistribution(NormalWeightedMeanPrecision, float([i, -i]))
    #         end
    #     end

    #     @testset "logpdf" begin
    #         for i in 1:10
    #             @test logpdf(ExponentialFamilyDistribution(NormalWeightedMeanPrecision, [i, -i]), 0) ≈
    #                   logpdf(NormalWeightedMeanPrecision(i, 2 * i), 0)
    #         end
    #     end

    #     @testset "isproper" begin
    #         for i in 1:10
    #             @test isproper(ExponentialFamilyDistribution(NormalWeightedMeanPrecision, [i, -i])) === true
    #             @test isproper(ExponentialFamilyDistribution(NormalWeightedMeanPrecision, [i, i])) === false
    #         end
    #     end

    #     @testset "fisherinformation" begin
    #         for (η1, η2) in Iterators.product(1:10, -10:1:-1)
    #             ef = ExponentialFamilyDistribution(NormalWeightedMeanPrecision, [η1, η2])
    #             f_logpartion = (η) -> logpartition(ExponentialFamilyDistribution(NormalWeightedMeanPrecision, η))
    #             autograd_inforamation_matrix = (η) -> ForwardDiff.hessian(f_logpartion, η)
    #             @test fisherinformation(ef) ≈ autograd_inforamation_matrix([η1, η2])
    #         end
    #     end
    # end

    # @testset "multivariate natural parameters related" begin
    #     @testset "Constructor" begin
    #         for i in 1:10
    #             @test convert(
    #                 Distribution,
    #                 ExponentialFamilyDistribution(MvGaussianWeightedMeanPrecision, vcat([i, 0], vec([-i 0; 0 -i])))
    #             ) ==
    #                   MvGaussianWeightedMeanPrecision([i, 0], [2*i 0; 0 2*i])

    #             @test convert(
    #                 ExponentialFamilyDistribution,
    #                 MvGaussianWeightedMeanPrecision([i, 0], [2*i 0; 0 2*i])
    #             ) ≈
    #                   ExponentialFamilyDistribution(
    #                 MvGaussianWeightedMeanPrecision,
    #                 vcat(float([i, 0]), float(vec([-i 0; 0 -i])))
    #             )
    #         end
    #     end

    #     @testset "logpdf" begin
    #         for i in 1:10
    #             mv_np = ExponentialFamilyDistribution(MvGaussianWeightedMeanPrecision, vcat([i, 0], vec([-i 0; 0 -i])))
    #             distribution = MvGaussianWeightedMeanPrecision([i, 0.0], [2*i -0.0; -0.0 2*i])
    #             @test logpdf(distribution, [0.0, 0.0]) ≈ logpdf(mv_np, [0.0, 0.0])
    #             @test logpdf(distribution, [1.0, 0.0]) ≈ logpdf(mv_np, [1.0, 0.0])
    #             @test logpdf(distribution, [1.0, 1.0]) ≈ logpdf(mv_np, [1.0, 1.0])
    #         end
    #     end

    #     @testset "logpartition" begin
    #         @test logpartition(ExponentialFamilyDistribution(NormalWeightedMeanPrecision, [1, -2])) ≈
    #               -(log(2) - 1 / 8)
    #     end

    #     @testset "isproper" begin
    #         for i in 1:10
    #             efproper = ExponentialFamilyDistribution(MvNormalMeanCovariance, vcat([i, 0], vec([-i 0; 0 -i])))
    #             efimproper = ExponentialFamilyDistribution(MvNormalMeanCovariance, vcat([i, 0], vec([i 0; 0 i])))
    #             @test isproper(efproper) === true
    #             @test isproper(efimproper) === false
    #         end
    #     end

    #     @testset "basemeasure" begin
    #         for i in 1:10
    #             @test basemeasure(
    #                 ExponentialFamilyDistribution(MvNormalMeanCovariance, vcat([i, 0], vec([-i 0; 0 -i]))),
    #                 rand(2)) == (2pi)^(-1)
    #         end
    #     end
    # end

    # @testset "fisherinformation" begin
    #     for (m, w) in Iterators.product(1:10, 1:10)
    #         dist = NormalMeanPrecision(m, w)
    #         normal_weighted_mean_precision = convert(NormalWeightedMeanPrecision, dist)
    #         J = [1/w -m/w; 0 1]
    #         @test J' * fisherinformation(dist) * J ≈ fisherinformation(normal_weighted_mean_precision)
    #     end
    # end

    # @testset "fisherinformation" begin
    #     rng = StableRNG(42)
    #     n_samples = 1000
    #     for (μ, var) in Iterators.product(-10:10, 0.5:0.5:10)
    #         samples = rand(rng, NormalMeanVariance(μ, var), n_samples)
    #         hessian_at_sample =
    #             (sample) ->
    #                 ForwardDiff.hessian((params) -> logpdf(NormalMeanVariance(params[1], params[2]), sample), [μ, var])
    #         expected_hessian = -mean(hessian_at_sample, samples)
    #         @test expected_hessian ≈ fisherinformation(NormalMeanVariance(μ, var)) atol = 0.5
    #     end
    # end

    # @testset "fisherinformation" begin
    #     for (xi, w) in Iterators.product(1:10, 1:10)
    #         dist = NormalWeightedMeanPrecision(xi, w)
    #         ef = convert(ExponentialFamilyDistribution, dist)
    #         J = [1 0; 0 -2]
    #         @test J * fisherinformation(dist) * J ≈ fisherinformation(ef)
    #     end
    # end

    # @testset "fisherinformation" begin
    #     function reconstructed_logpartition(::ExponentialFamilyDistribution{T}, ηvec) where {T}
    #         return logpartition(ExponentialFamilyDistribution(T, ηvec))
    #     end

    #     function transformation(ef::ExponentialFamilyDistribution{T}, ηvec) where {T}
    #         natural_params = getnaturalparameters(ef)
    #         mean_size = length(natural_params[1])
    #         @views wmean = ηvec[1:mean_size]
    #         @views matrix = reshape(ηvec[(mean_size+1):end], mean_size, mean_size)
    #         ef = ExponentialFamilyDistribution(T, [wmean, matrix])
    #         mean_cov = mean(ef), cov(ef)
    #         return [mean_cov[1]..., mean_cov[2]...]
    #     end

    #     # Autograd friendly version of Gaussian logpdf
    #     function friendlygaussianlpdf(params, x)
    #         k = length(x)
    #         μ, Σ = params[1:k], reshape(params[k+1:end], k, k)
    #         coef = (2π)^(-k / 2) * det(Σ)^(-1 / 2)
    #         exponent = -0.5 * (x - μ)' * inv(Σ) * (x - μ)
    #         return log(coef * exp(exponent))
    #     end

    #     # We test Gaussian up to size 3 as autograd gets very slow for anything larger than that
    #     rng = StableRNG(42)
    #     for i in 1:5, d in 2:3
    #         μ = rand(rng, d)
    #         L = randn(rng, d, d)
    #         Σ = L * L'
    #         dist = MvNormalMeanCovariance(μ, Σ)
    #         ef = convert(ExponentialFamilyDistribution, dist)
    #         v = getnaturalparameters(ef)
    #         fi_ag = ForwardDiff.hessian(x -> reconstructed_logpartition(ef, x), v)
    #         # WARNING: ForwardDiff returns a non-positive definite Hessian for a convex function. 
    #         # The matrices are identical up to permutations, resulting in eigenvalues that are the same up to a sign.
    #         fi_ef = fisherinformation(ef)
    #         @test sort(eigvals(fi_ef)) ≈ sort(abs.(eigvals(fi_ag))) atol = 1e-6
    #     end

    #     # We normally perform test with jacobian transformation, but autograd fails to compute jacobians with duplicated elements.
    #     # For d > 2 the error becomes larger, so we set d = 2
    #     for i in 1:5, d in 2:2
    #         μ = rand(rng, d)
    #         L = randn(rng, d, d)
    #         Σ = L * L'
    #         n_samples = 100000
    #         dist = MvNormalMeanCovariance(μ, Σ)

    #         samples = rand(rng, dist, n_samples)
    #         samples = [samples[:, i] for i in 1:n_samples]

    #         v_ = [μ..., as_vec(Σ)...]
    #         totalHessian = zeros(length(v_), length(v_))
    #         for sample in samples
    #             totalHessian -= ForwardDiff.hessian(x -> friendlygaussianlpdf(x, sample), v_)
    #         end
    #         computed_hessian = totalHessian /= n_samples

    #         # The error will be higher for sampling tests, tolerance adjusted accordingly.
    #         fi_dist = fisherinformation(dist)
    #         @test sort(eigvals(fi_dist)) ≈ sort(abs.(eigvals(computed_hessian))) rtol = 1e-1
    #         @test sort(svd(fi_dist).S) ≈ sort(svd(computed_hessian).S) rtol = 1e-1
    #     end
    # end
end

end
