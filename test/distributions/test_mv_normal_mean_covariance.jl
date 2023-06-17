module MvNormalMeanCovarianceTest

using Test
using ExponentialFamily
using LinearAlgebra
using Distributions
using ForwardDiff
using StableRNGs

import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, fisherinformation,as_vec

@testset "MvNormalMeanCovariance" begin
    @testset "Constructor" begin
        @test MvNormalMeanCovariance <: AbstractMvNormal

        @test MvNormalMeanCovariance([1.0, 1.0]) == MvNormalMeanCovariance([1.0, 1.0], [1.0, 1.0])
        @test MvNormalMeanCovariance([1.0, 2.0]) == MvNormalMeanCovariance([1.0, 2.0], [1.0, 1.0])
        @test MvNormalMeanCovariance([1, 2]) == MvNormalMeanCovariance([1.0, 2.0], [1.0, 1.0])
        @test MvNormalMeanCovariance([1.0f0, 2.0f0]) == MvNormalMeanCovariance([1.0f0, 2.0f0], [1.0f0, 1.0f0])

        @test eltype(MvNormalMeanCovariance([1.0, 1.0])) === Float64
        @test eltype(MvNormalMeanCovariance([1.0, 1.0], [1.0, 1.0])) === Float64
        @test eltype(MvNormalMeanCovariance([1, 1])) === Float64
        @test eltype(MvNormalMeanCovariance([1, 1], [1, 1])) === Float64
        @test eltype(MvNormalMeanCovariance([1.0f0, 1.0f0])) === Float32
        @test eltype(MvNormalMeanCovariance([1.0f0, 1.0f0], [1.0f0, 1.0f0])) === Float32
    end

    @testset "distrname" begin
        @test ExponentialFamily.distrname(MvNormalMeanCovariance(zeros(2))) === "MvNormalMeanCovariance"
    end

    @testset "Stats methods" begin
        μ    = [0.2, 3.0, 4.0]
        Σ    = [1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5]
        dist = MvNormalMeanCovariance(μ, Σ)

        @test mean(dist) == μ
        @test mode(dist) == μ
        @test weightedmean(dist) ≈ cholinv(Σ) * μ
        @test invcov(dist) ≈ cholinv(Σ)
        @test precision(dist) ≈ cholinv(Σ)
        @test cov(dist) == Σ
        @test std(dist) ≈ cholsqrt(Σ)
        @test all(mean_cov(dist) .≈ (μ, Σ))
        @test all(mean_invcov(dist) .≈ (μ, cholinv(Σ)))
        @test all(mean_precision(dist) .≈ (μ, cholinv(Σ)))
        @test all(weightedmean_cov(dist) .≈ (cholinv(Σ) * μ, Σ))
        @test all(weightedmean_invcov(dist) .≈ (cholinv(Σ) * μ, cholinv(Σ)))
        @test all(weightedmean_precision(dist) .≈ (cholinv(Σ) * μ, cholinv(Σ)))

        @test length(dist) == 3
        @test entropy(dist) ≈ 5.361886000915401
        @test pdf(dist, [0.2, 3.0, 4.0]) ≈ 0.021028302702542
        @test pdf(dist, [0.202, 3.002, 4.002]) ≈ 0.021028229679079503
        @test logpdf(dist, [0.2, 3.0, 4.0]) ≈ -3.8618860009154012
        @test logpdf(dist, [0.202, 3.002, 4.002]) ≈ -3.861889473548943
    end

    @testset "Base methods" begin
        @test convert(MvNormalMeanCovariance{Float32}, MvNormalMeanCovariance([0.0, 0.0])) ==
              MvNormalMeanCovariance([0.0f0, 0.0f0], [1.0f0, 1.0f0])
        @test convert(MvNormalMeanCovariance{Float64}, [0.0, 0.0], [2 0; 0 3]) ==
              MvNormalMeanCovariance([0.0, 0.0], [2.0 0.0; 0.0 3.0])

        @test length(MvNormalMeanCovariance([0.0, 0.0])) === 2
        @test length(MvNormalMeanCovariance([0.0, 0.0, 0.0])) === 3
        @test ndims(MvNormalMeanCovariance([0.0, 0.0])) === 2
        @test ndims(MvNormalMeanCovariance([0.0, 0.0, 0.0])) === 3
        @test size(MvNormalMeanCovariance([0.0, 0.0])) === (2,)
        @test size(MvNormalMeanCovariance([0.0, 0.0, 0.0])) === (3,)
    end

    @testset "vague" begin
        @test_throws MethodError vague(MvNormalMeanCovariance)

        d1 = vague(MvNormalMeanCovariance, 2)

        @test typeof(d1) <: MvNormalMeanCovariance
        @test mean(d1) == zeros(2)
        @test cov(d1) == Matrix(Diagonal(1e12 * ones(2)))
        @test ndims(d1) == 2

        d2 = vague(MvNormalMeanCovariance, 3)

        @test typeof(d2) <: MvNormalMeanCovariance
        @test mean(d2) == zeros(3)
        @test cov(d2) == Matrix(Diagonal(1e12 * ones(3)))
        @test ndims(d2) == 3
    end

    @testset "prod" begin
        @test prod(ClosedProd(), MvNormalMeanCovariance([-1, -1], [2, 2]), MvNormalMeanCovariance([1, 1], [2, 4])) ≈
              MvNormalWeightedMeanPrecision([0, -1 / 4], [1, 3 / 4])

        μ    = [1.0, 2.0, 3.0]
        Σ    = diagm([1.0, 2.0, 3.0])
        dist = MvNormalMeanCovariance(μ, Σ)

        @test prod(ClosedProd(), dist, dist) ≈
              MvNormalWeightedMeanPrecision([2.0, 2.0, 2.0], diagm([2.0, 1.0, 2 / 3]))
    end

    @testset "convert" begin
        @test convert(MvNormalMeanCovariance, zeros(2), Matrix(Diagonal(ones(2)))) ==
              MvNormalMeanCovariance(zeros(2), Matrix(Diagonal(ones(2))))
        @test begin
            m = rand(5)
            c = Matrix(Symmetric(rand(5, 5)))
            convert(MvNormalMeanCovariance, m, c) == MvNormalMeanCovariance(m, c)
        end
    end

    @testset "fisherinformation" begin
        function reconstructed_logpartition(ef::KnownExponentialFamilyDistribution{T}, ηvec) where {T}
            natural_params = getnaturalparameters(ef)
            mean_size = length(natural_params[1])
            @views wmean = ηvec[1:mean_size]
            @views matrix = reshape(ηvec[(mean_size+1):end], mean_size, mean_size)
            ef = KnownExponentialFamilyDistribution(T, [wmean, matrix])
            return logpartition(ef)
        end

        function transformation(ef::KnownExponentialFamilyDistribution{T}, ηvec) where {T}
            natural_params = getnaturalparameters(ef)
            mean_size = length(natural_params[1])
            @views wmean = ηvec[1:mean_size]
            @views matrix = reshape(ηvec[(mean_size+1):end], mean_size, mean_size)
            ef = KnownExponentialFamilyDistribution(T, [wmean, matrix])
            mean_cov = mean(ef), cov(ef)
            return [mean_cov[1]..., mean_cov[2]...]
        end

        # Autograd friendly version of Gaussian logpdf
        function friendlygaussianlpdf(params, x)
            k = length(x)
            μ, Σ = params[1:k], reshape(params[k+1:end], k, k)
            coef = (2π)^(-k / 2) * det(Σ)^(-1 / 2)
            exponent = -0.5 * (x - μ)' * inv(Σ) * (x - μ)
            return log(coef * exp(exponent))
        end

        # We test Gaussian up to size 3 as autograd gets very slow for anything larger than that
        rng = StableRNG(42)
        for i in 1:5, d in 2:3
            μ = rand(rng, d)
            L = randn(rng, d, d)
            Σ = L * L'
            dist = MvNormalMeanCovariance(μ, Σ)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            v = [getnaturalparameters(ef)[1]..., getnaturalparameters(ef)[2]...]
            fi_ag = ForwardDiff.hessian(x -> reconstructed_logpartition(ef, x), v)
            # WARNING: ForwardDiff returns a non-positive definite Hessian for a convex function. 
            # The matrices are identical up to permutations, resulting in eigenvalues that are the same up to a sign.
            fi_ef = fisherinformation(ef)
            @test sort(eigvals(fi_ef)) ≈ sort(abs.(eigvals(fi_ag))) atol = 1e-6
        end

        # We normally perform test with jacobian transformation, but autograd fails to compute jacobians with duplicated elements.
        # For d > 2 the error becomes larger, so we set d = 2
        for i in 1:5, d in 2:2
            μ = rand(rng, d)
            L = randn(rng, d, d)
            Σ = L * L'
            n_samples = 100000
            dist = MvNormalMeanCovariance(μ, Σ)

            samples = rand(rng, dist, n_samples)
            samples = [samples[:, i] for i in 1:n_samples]

            v_ = [μ..., as_vec(Σ)...]
            totalHessian = zeros(length(v_), length(v_))
            for sample in samples
                totalHessian -= ForwardDiff.hessian(x -> friendlygaussianlpdf(x, sample), v_)
            end
            computed_hessian = totalHessian /= n_samples

            # The error will be higher for sampling tests, tolerance adjusted accordingly.
            fi_dist = fisherinformation(dist)
            @test sort(eigvals(fi_dist)) ≈ sort(abs.(eigvals(computed_hessian))) rtol = 1e-1
            @test sort(svd(fi_dist).S) ≈ sort(svd(computed_hessian).S) rtol = 1e-1
        end
    end
end

end
