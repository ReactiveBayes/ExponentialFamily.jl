
@testitem "MvNormalMeanScalePrecision: Constructor" begin
    include("./normal_family_setuptests.jl")

    @test MvNormalMeanScalePrecision <: AbstractMvNormal

    @test MvNormalMeanScalePrecision([1.0, 1.0]) == MvNormalMeanScalePrecision([1.0, 1.0], 1.0)
    @test MvNormalMeanScalePrecision([1.0, 2.0]) == MvNormalMeanScalePrecision([1.0, 2.0], 1.0)
    @test MvNormalMeanScalePrecision([1, 2]) == MvNormalMeanScalePrecision([1.0, 2.0], 1.0)
    @test MvNormalMeanScalePrecision([1.0f0, 2.0f0]) == MvNormalMeanScalePrecision([1.0f0, 2.0f0], 1.0f0)

    @test eltype(MvNormalMeanScalePrecision([1.0, 1.0])) === Float64
    @test eltype(MvNormalMeanScalePrecision([1.0, 1.0], 1.0)) === Float64
    @test eltype(MvNormalMeanScalePrecision([1, 1])) === Float64
    @test eltype(MvNormalMeanScalePrecision([1, 1], 1)) === Float64
    @test eltype(MvNormalMeanScalePrecision([1.0f0, 1.0f0])) === Float32
    @test eltype(MvNormalMeanScalePrecision([1.0f0, 1.0f0], 1.0f0)) === Float32

    @test MvNormalMeanScalePrecision(ones(3), 5) == MvNormalMeanScalePrecision(ones(3), 5)
    @test MvNormalMeanScalePrecision([1, 2, 3, 4], 7.0) == MvNormalMeanScalePrecision([1.0, 2.0, 3.0, 4.0], 7.0)
end

@testitem "MvNormalMeanScalePrecision: distrname" begin
    include("./normal_family_setuptests.jl")

    @test ExponentialFamily.distrname(MvNormalMeanScalePrecision(zeros(2))) === "MvNormalMeanScalePrecision"
end

@testitem "MvNormalMeanScalePrecision: ExponentialFamilyDistribution" begin
    include("../distributions_setuptests.jl")

    rng = StableRNG(42)

    for s in 1:6
        μ = randn(rng, s)
        γ = rand(rng)

        @testset let d = MvNormalMeanScalePrecision(μ, γ)
            ef = test_exponentialfamily_interface(d;)
        end
    end

    μ = randn(rng, 1)
    γ = rand(rng)

    d = MvNormalMeanScalePrecision(μ, γ)
    ef = convert(ExponentialFamilyDistribution, d)

    d1d = NormalMeanPrecision(μ[1], γ)
    ef1d = convert(ExponentialFamilyDistribution, d1d)

    @test logpartition(ef) ≈ logpartition(ef1d)
    @test gradlogpartition(ef) ≈ gradlogpartition(ef1d)
    @test fisherinformation(ef) ≈ fisherinformation(ef1d)
end

@testitem "MvNormalMeanScalePrecision: Stats methods" begin
    include("./normal_family_setuptests.jl")

    μ = [0.2, 3.0, 4.0]
    γ = 2.0
    dist = MvNormalMeanScalePrecision(μ, γ)
    rdist = MvNormalMeanPrecision(μ, γ * ones(length(μ)))

    @test mean(dist) == μ
    @test mode(dist) == μ
    @test scale(dist) == γ
    @test weightedmean(dist) == weightedmean(rdist)
    @test invcov(dist) == invcov(rdist)
    @test precision(dist) == precision(rdist)
    @test cov(dist) ≈ cov(rdist)
    @test std(dist) * std(dist)' ≈ std(rdist) * std(rdist)'
    @test all(mean_cov(dist) .≈ mean_cov(rdist))
    @test all(mean_invcov(dist) .≈ mean_invcov(rdist))
    @test all(mean_precision(dist) .≈ mean_precision(rdist))
    @test all(weightedmean_cov(dist) .≈ weightedmean_cov(rdist))
    @test all(weightedmean_invcov(dist) .≈ weightedmean_invcov(rdist))
    @test all(weightedmean_precision(dist) .≈ weightedmean_precision(rdist))

    @test length(dist) == 3
    @test entropy(dist) ≈ entropy(rdist)
    @test pdf(dist, [0.2, 3.0, 4.0]) ≈ pdf(rdist, [0.2, 3.0, 4.0])
    @test pdf(dist, [0.202, 3.002, 4.002]) ≈ pdf(rdist, [0.202, 3.002, 4.002]) atol = 1e-4
    @test logpdf(dist, [0.2, 3.0, 4.0]) ≈ logpdf(rdist, [0.2, 3.0, 4.0])
    @test logpdf(dist, [0.202, 3.002, 4.002]) ≈ logpdf(rdist, [0.202, 3.002, 4.002]) atol = 1e-4
    @test rand(StableRNG(42), dist, 1000) ≈ rand(StableRNG(42), rdist, 1000)
end

@testitem "MvNormalMeanScalePrecision: Base methods" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalMeanScalePrecision{Float32}, MvNormalMeanScalePrecision([0.0, 0.0])) ==
          MvNormalMeanScalePrecision([0.0f0, 0.0f0], 1.0f0)
    @test convert(MvNormalMeanScalePrecision{Float64}, [0.0, 0.0], 2.0) ==
          MvNormalMeanScalePrecision([0.0, 0.0], 2.0)

    @test length(MvNormalMeanScalePrecision([0.0, 0.0])) === 2
    @test length(MvNormalMeanScalePrecision([0.0, 0.0, 0.0])) === 3
    @test ndims(MvNormalMeanScalePrecision([0.0, 0.0])) === 2
    @test ndims(MvNormalMeanScalePrecision([0.0, 0.0, 0.0])) === 3
    @test size(MvNormalMeanScalePrecision([0.0, 0.0])) === (2,)
    @test size(MvNormalMeanScalePrecision([0.0, 0.0, 0.0])) === (3,)

    μ, γ = zeros(2), 2.0
    distribution = MvNormalMeanScalePrecision(μ, γ)

    @test distribution ≈ distribution
    @test convert(MvNormalMeanCovariance, distribution) == MvNormalMeanCovariance(μ, inv(γ) * I(length(μ)))
    @test convert(MvNormalMeanPrecision, distribution) == MvNormalMeanPrecision(μ, γ * I(length(μ)))
    @test convert(MvNormalWeightedMeanPrecision, distribution) == MvNormalWeightedMeanPrecision(γ * μ, γ * I(length(μ)))
end

@testitem "MvNormalMeanScalePrecision: vague" begin
    include("./normal_family_setuptests.jl")

    @test_throws MethodError vague(MvNormalMeanScalePrecision)

    d1 = vague(MvNormalMeanScalePrecision, 2)

    @test typeof(d1) <: MvNormalMeanScalePrecision
    @test mean(d1) == zeros(2)
    @test invcov(d1) == Matrix(Diagonal(1e-12 * ones(2)))
    @test ndims(d1) == 2

    d2 = vague(MvNormalMeanScalePrecision, 3)

    @test typeof(d2) <: MvNormalMeanScalePrecision
    @test mean(d2) == zeros(3)
    @test invcov(d2) == Matrix(Diagonal(1e-12 * ones(3)))
    @test ndims(d2) == 3
end

@testitem "MvNormalMeanScalePrecision: prod" begin
    include("./normal_family_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
        @test prod(strategy, MvNormalMeanScalePrecision([-1, -1], 2), MvNormalMeanPrecision([1, 1], [2, 4])) ≈
              MvNormalWeightedMeanPrecision([0, 2], [4, 6])

        μ    = [1.0, 2.0, 3.0]
        γ    = 2.0
        dist = MvNormalMeanScalePrecision(μ, γ)

        @test prod(strategy, dist, dist) ≈
              MvNormalMeanScalePrecision([1.0, 2.0, 3.0], 2γ)
    end
end

@testitem "MvNormalMeanScalePrecision: convert" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalMeanScalePrecision, zeros(2), 1.0) ==
          MvNormalMeanScalePrecision(zeros(2), 1.0)
    @test begin
        m = rand(5)
        c = rand()
        convert(MvNormalMeanScalePrecision, m, c) == MvNormalMeanScalePrecision(m, c)
    end
end

@testitem "MvNormalMeanScalePrecision: rand" begin
    include("./normal_family_setuptests.jl")

    rng = MersenneTwister(42)

    for T in (Float32, Float64)
        @testset "Basic functionality" begin
            μ = [1.0, 2.0, 3.0]
            γ = 2.0
            dist = convert(MvNormalMeanScalePrecision{T}, μ, γ)

            @test typeof(rand(dist)) <: Vector{T}

            samples = rand(rng, dist, 5_000)

            @test isapprox(mean(samples), mean(μ), atol = 0.5)
        end
    end
end

@testitem "MvNormalMeanScalePrecision: Fisher is faster then for full parametrization" begin
    include("./normal_family_setuptests.jl")

    using BenchmarkTools
    using FastCholesky
    using LinearAlgebra
    using JET

    rng = StableRNG(42)
    for k in 10:5:40
        μ = randn(rng, k)
        γ = rand(rng)
        cov = inv(γ) * I(k)

        ef_small = convert(ExponentialFamilyDistribution, MvNormalMeanScalePrecision(μ, γ))
        ef_full = convert(ExponentialFamilyDistribution, MvNormalMeanCovariance(μ, cov))

        fi_small = fisherinformation(ef_small)
        fi_full = fisherinformation(ef_full)

        benchmark_fi_mvsp = @benchmark fisherinformation($ef_small) seconds = 1
        benchmark_fi_full = @benchmark fisherinformation($ef_full) seconds = 1

        @test_opt cholinv(fi_small)
        @test_opt cholinv(fi_full)

        # `cholinv` for the small returns a wrapper and doesn't really compute the inverse
        # but a proxy to compute products, the conversion to Matrix will display a warning, but its oke
        b = randn(rng, k + 1)
        Base.with_logger(Base.NullLogger()) do
            @test cholinv(fi_small) \ b ≈ cholinv(Matrix(fi_small)) \ b
        end

        benchmark_cholinv_mvsp = @benchmark cholinv($fi_small) seconds = 1
        benchmark_cholinv_full = @benchmark cholinv($fi_full) seconds = 1

        # small time is supposed to be O(k) and full time is supposed to O(k^2)
        # small time is supposed to be O(k) and full time is supposed to O(k^2)
        # the constant C is selected to account to fluctuations in test runs
        C = 0.7
        @test mean(benchmark_fi_mvsp.times) < mean(benchmark_fi_full.times) / (C * k)
        @test benchmark_fi_mvsp.allocs < benchmark_fi_full.allocs

        @test mean(benchmark_cholinv_mvsp.times) < mean(benchmark_cholinv_full.times) / (C * k)
        @test benchmark_cholinv_mvsp.allocs < benchmark_cholinv_full.allocs
    end
end
