module NormalTest

using ExponentialFamily, Distributions
using Test, LinearAlgebra, ForwardDiff, Random, StableRNGs, SparseArrays

include("../../testutils.jl")

import ExponentialFamily: promote_variate_type, fastcholesky

# We need this extra function to ensure better derivatives with AD, it is slower than our implementation
# but is more AD friendly
function getlogpartitionfortest(::NaturalParametersSpace, ::Type{MvNormalMeanCovariance})
    return (η) -> begin
        weightedmean, minushalfprecision = unpack_parameters(MvNormalMeanCovariance, η)
        return (dot(weightedmean, inv(-minushalfprecision), weightedmean) / 2 - logdet(-2 * minushalfprecision)) / 2
    end
end

function gaussianlpdffortest(params, x)
    k = length(x)
    μ, Σ = params[1:k], reshape(params[k+1:end], k, k)
    coef = (2π)^(-k / 2) * det(Σ)^(-1 / 2)
    exponent = -0.5 * (x - μ)' * inv(Σ) * (x - μ)
    return log(coef * exp(exponent))
end

function check_basic_statistics(left::UnivariateNormalDistributionsFamily, right::UnivariateNormalDistributionsFamily)
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

    # `Normal` is not defining some of these methods and we don't want to define them either, because of the type piracy
    if !(left isa Normal || right isa Normal)
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

function check_basic_statistics(left::MultivariateNormalDistributionsFamily, right::MultivariateNormalDistributionsFamily)
    @test mean(left) ≈ mean(right)
    @test mode(left) ≈ mode(right)
    @test var(left) ≈ var(right)
    @test cov(left) ≈ cov(right)
    @test logdetcov(left) ≈ logdetcov(right)
    @test length(left) === length(right)
    @test size(left) === size(right)
    @test entropy(left) ≈ entropy(right)

    dims = length(mean(left))

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
                atol = 1e-12
            )
        )
        @test all(
            isapprox.(
                ForwardDiff.hessian((x) -> logpdf(left, x), value),
                ForwardDiff.hessian((x) -> logpdf(right, x), value),
                atol = 1e-12
            )
        )
    end

    # `MvNormal` is not defining some of these methods and we don't want to define them either, because of the type piracy
    if !(left isa MvNormal || right isa MvNormal)
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

@testset "Normal family" begin
    @testset "Univariate conversions" begin
        types  = ExponentialFamily.union_types(UnivariateNormalDistributionsFamily{Float64})
        etypes = ExponentialFamily.union_types(UnivariateNormalDistributionsFamily)

        rng = MersenneTwister(1234)

        for type in types
            left = convert(type, rand(rng, Float64), rand(rng, Float64))

            for type in [types..., etypes...]
                right = convert(type, left)
                check_basic_statistics(left, right)

                p1 = prod(PreserveTypeLeftProd(), left, right)
                @test typeof(p1) <: typeof(left)

                p2 = prod(PreserveTypeRightProd(), left, right)
                @test typeof(p2) <: typeof(right)

                for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
                    p3 = prod(strategy, left, right)

                    check_basic_statistics(p1, p2)
                    check_basic_statistics(p2, p3)
                    check_basic_statistics(p1, p3)
                end
            end
        end
    end

    @testset "Multivariate conversions" begin
        types  = ExponentialFamily.union_types(MultivariateNormalDistributionsFamily{Float64})
        etypes = ExponentialFamily.union_types(MultivariateNormalDistributionsFamily)

        dims = (2, 3, 5)
        rng  = MersenneTwister(1234)

        for dim in dims
            for type in types
                left = convert(type, rand(rng, Float64, dim), Matrix(Diagonal(abs.(rand(rng, Float64, dim)))))

                for type in [types..., etypes...]
                    right = convert(type, left)
                    check_basic_statistics(left, right)

                    p1 = prod(PreserveTypeLeftProd(), left, right)
                    @test typeof(p1) <: typeof(left)

                    p2 = prod(PreserveTypeRightProd(), left, right)
                    @test typeof(p2) <: typeof(right)

                    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
                        p3 = prod(strategy, left, right)

                        check_basic_statistics(p1, p2)
                        check_basic_statistics(p2, p3)
                        check_basic_statistics(p1, p3)
                    end
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

    @testset "ExponentialFamilyDistribution{NormalMeanVariance}" begin
        @testset for μ in -10.0:5.0:10.0, σ² in 0.1:1.0:5.0, T in ExponentialFamily.union_types(UnivariateNormalDistributionsFamily)
            @testset let d = convert(T, NormalMeanVariance(μ, σ²))
                ef = test_exponentialfamily_interface(d)

                (η₁, η₂) = (mean(d) / var(d), -1 / 2var(d))

                for x in 10randn(4)
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) ≈ 1 / sqrt(2π)
                    @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x, abs2(x)))
                    @test @inferred(logpartition(ef)) ≈ (-η₁^2 / 4η₂ - 1 / 2 * log(-2η₂))
                    @test @inferred(insupport(ef, x))
                end
            end
        end

        # Test failing isproper cases
        @test !isproper(MeanParametersSpace(), NormalMeanVariance, [-1])
        @test !isproper(MeanParametersSpace(), NormalMeanVariance, [1, -0.1])
        @test !isproper(MeanParametersSpace(), NormalMeanVariance, [-0.1, -1])
        @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [-1.1])
        @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [1, 1])
        @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [-1.1, 1])
    end

    @testset "prod with ExponentialFamilyDistribution{NormalMeanVariance}" for μleft in 10randn(4), σ²left in 10rand(4), μright in 10randn(4),
        σ²right in 10rand(4), Tleft in ExponentialFamily.union_types(UnivariateNormalDistributionsFamily),
        Tright in ExponentialFamily.union_types(UnivariateNormalDistributionsFamily)

        @testset let (left, right) = (convert(Tleft, NormalMeanVariance(μleft, σ²left)), convert(Tright, NormalMeanVariance(μright, σ²right)))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{NormalMeanVariance})
                )
            )
        end
    end

    @testset "ExponentialFamilyDistribution{MvNormalMeanCovariance}" begin
        @testset for s in (2, 3), T in ExponentialFamily.union_types(MultivariateNormalDistributionsFamily)
            μ = 10randn(s)
            L = randn(s, s)
            Σ = L * L'
            @testset let d = convert(T, MvNormalMeanCovariance(μ, Σ))
                ef = test_exponentialfamily_interface(
                    d;
                    # These are handled differently below
                    test_fisherinformation_against_hessian = false,
                    test_fisherinformation_against_jacobian = false
                )

                (η₁, η₂) = (cholinv(Σ) * mean(d), -cholinv(Σ) / 2)

                for x in [10randn(s) for _ in 1:4]
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) ≈ (2π)^(-s / 2)
                    @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x, x * x'))
                    @test @inferred(logpartition(ef)) ≈ -1 / 4 * (η₁' * inv(η₂) * η₁) - 1 / 2 * logdet(-2η₂)
                    @test @inferred(insupport(ef, x))
                end
            end
        end

        # Test failing isproper cases (naive)
        @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [-1])
        @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [1, -0.1])
        @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [-0.1, -1])
        @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [-1, 2, 3, 4]) # shapes are incompatible
        @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [1, -0.1, -1, 0, 0, -1]) # covariance is not posdef

        @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [-1.1])
        @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [1, 1])
        @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [-1.1, 1])
        @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [-1, 2, 3, 4]) # shapes are incompatible
        @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [1, -0.1, 1, 0, 0, 1]) # -η₂ is not posdef
    end

    @testset "Fisher information matrix in natural parameters space" for i in 1:5, d in 2:10
        rng = StableRNG(d * i)
        μ = rand(rng, d)
        L = randn(rng, d, d)
        Σ = L * L'
        ef = convert(ExponentialFamilyDistribution, MvNormalMeanCovariance(μ, Σ))

        fi_ef = fisherinformation(ef)
        # @test_broken isposdef(fi_ef)
        # The `isposdef` check is not really reliable in Julia, here, instead
        # we compute eigen values and additionally check that our `fastcholesky` inverse actually produces the correct inverse
        @test issymmetric(fi_ef) || (LowerTriangular(fi_ef) ≈ (UpperTriangular(fi_ef)'))
        @test isposdef(fi_ef) || all(>(0), eigvals(fi_ef))

        fi_ef_inv = inv(fastcholesky(fi_ef))
        @test (fi_ef_inv * fi_ef) ≈ Diagonal(ones(d + d^2)) rtol = 1e-2

        # WARNING: ForwardDiff returns a non-positive definite Hessian for a convex function. 
        # The matrices are identical up to permutations, resulting in eigenvalues that are the same up to a sign.
        fi_ag = ForwardDiff.hessian(getlogpartitionfortest(NaturalParametersSpace(), MvNormalMeanCovariance), getnaturalparameters(ef))
        @test norm(sort(eigvals(fi_ef)) - sort(abs.(eigvals(fi_ag)))) ≈ 0 atol = (6e-5 * d^2)
    end

    # We normally perform test with jacobian transformation, but autograd fails to compute jacobians with duplicated elements.
    @testset "Fisher information matrix in mean parameters space" for i in 1:5, d in 2:3
        rng = StableRNG(d * i)
        μ = rand(rng, d)
        L = randn(rng, d, d)
        Σ = L * L'
        n_samples = 10000
        dist = MvNormalMeanCovariance(μ, Σ)

        samples = rand(rng, dist, n_samples)

        θ = pack_parameters(MvNormalMeanCovariance, (μ, Σ))

        approxHessian = zeros(length(θ), length(θ))
        for sample in eachcol(samples)
            approxHessian -= ForwardDiff.hessian(Base.Fix2(gaussianlpdffortest, sample), θ)
        end
        approxFisherInformation = approxHessian /= n_samples

        # The error will be higher for sampling tests, tolerance adjusted accordingly.
        fi_dist = getfisherinformation(MeanParametersSpace(), MvNormalMeanCovariance)(θ)
        @test isposdef(fi_dist) || all(>(0), eigvals(fi_dist))
        @test issymmetric(fi_dist) || (LowerTriangular(fi_dist) ≈ (UpperTriangular(fi_dist)'))
        @test sort(eigvals(fi_dist)) ≈ sort(abs.(eigvals(approxFisherInformation))) rtol = 1e-1
        @test sort(svd(fi_dist).S) ≈ sort(svd(approxFisherInformation).S) rtol = 1e-1
    end
end

end
