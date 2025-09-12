
@testitem "MvNormalMeanScaleMatrixPrecision: Constructor" begin
    include("./normal_family_setuptests.jl")
    using LinearAlgebra

    @test MvNormalMeanScaleMatrixPrecision <: AbstractMvNormal

    @test MvNormalMeanScaleMatrixPrecision([1.0, 1.0]) == MvNormalMeanScaleMatrixPrecision([1.0, 1.0], 1.0, Matrix(I,2,2))
    @test MvNormalMeanScaleMatrixPrecision([1.0, 2.0]) == MvNormalMeanScaleMatrixPrecision([1.0, 2.0], 1.0, Matrix(I,2,2))
    @test MvNormalMeanScaleMatrixPrecision([1, 2]) == MvNormalMeanScaleMatrixPrecision([1.0, 2.0], 1.0, Matrix(I,2,2))
    @test MvNormalMeanScaleMatrixPrecision([1.0f0, 2.0f0]) == MvNormalMeanScaleMatrixPrecision([1.0f0, 2.0f0], 1.0f0, Matrix(I,2,2))

    @test eltype(MvNormalMeanScaleMatrixPrecision([1.0, 1.0])) === Float64
    @test eltype(MvNormalMeanScaleMatrixPrecision([1.0, 1.0], 1.0)) === Float64
    @test eltype(MvNormalMeanScaleMatrixPrecision([1, 1])) === Float64
    @test eltype(MvNormalMeanScaleMatrixPrecision([1, 1], 1)) === Float64
    @test eltype(MvNormalMeanScaleMatrixPrecision([1.0f0, 1.0f0])) === Float32
    @test eltype(MvNormalMeanScaleMatrixPrecision([1.0f0, 1.0f0], 1.0f0)) === Float32

    @test MvNormalMeanScaleMatrixPrecision(ones(3), 5, Matrix(I,3,3)) == MvNormalMeanScaleMatrixPrecision(ones(3), 5, Matrix(I,3,3))
    @test MvNormalMeanScaleMatrixPrecision([1, 2, 3, 4], 7.0, Matrix(I,4,4)) == MvNormalMeanScaleMatrixPrecision([1.0, 2.0, 3.0, 4.0], 7.0, Matrix(I,4,4))
end

@testitem "MvNormalMeanScaleMatrixPrecision: distrname" begin
    include("./normal_family_setuptests.jl")

    @test ExponentialFamily.distrname(MvNormalMeanScaleMatrixPrecision(zeros(2))) === "MvNormalMeanPrecision"
end

@testitem "MvNormalMeanScaleMatrixPrecision: is alias to MvNormalMeanPrecision" begin
    include("../distributions_setuptests.jl")

    rng = StableRNG(42)
    # integer works!
    @test typeof(MvNormalMeanScaleMatrixPrecision([1], 1, [1;;])) <: MvNormalMeanPrecision

    for s in 1:6
        μ = randn(rng, s)
        γ = rand(rng)
        g = randn(rng, s, s)
        G = g*g' + 1e-8*Matrix(I,s,s) # make sure it's pos definite

        @test typeof(MvNormalMeanScaleMatrixPrecision(μ, γ, G)) <: MvNormalMeanPrecision
    end
end

@testitem "MvNormalMeanScaleMatrixPrecision: ExponentialFamilyDistribution" begin
    include("../distributions_setuptests.jl")

    rng = StableRNG(42)

    for s in 1:6
        μ = randn(rng, s)
        γ = rand(rng)
        g = randn(rng, s, s)
        G = g*g' + 1e-8*Matrix(I,s,s) # make sure it's pos definite

        @testset let d = MvNormalMeanScaleMatrixPrecision(μ, γ, G)
            ef = test_exponentialfamily_interface(d;)
        end
    end

    μ = randn(rng, 1)
    γ = rand(rng)
    g = randn(rng, 1, 1)
    G = g*g' + 1e-8*Matrix(I,1,1) # make sure it's pos definite

    d = MvNormalMeanScaleMatrixPrecision(μ, γ, G)
    ef = convert(ExponentialFamilyDistribution, d)

    d1d = NormalMeanPrecision(μ, γ*G)
    ef1d = convert(ExponentialFamilyDistribution, d1d)

    @test logpartition(ef) ≈ logpartition(ef1d)
    @test gradlogpartition(ef) ≈ gradlogpartition(ef1d)
    @test fisherinformation(ef) ≈ fisherinformation(ef1d)
end

@testitem "MvNormalMeanScaleMatrixPrecision: Stats methods" begin
    include("./normal_family_setuptests.jl")
    
    rng = StableRNG(42)

    μ = [0.2, 3.0, 4.0]
    γ = 2.0
    g = randn(rng, 3, 3)
    G = g*g' + 1e-8*Matrix(I,3,3) # make sure it's pos definite
    dist = MvNormalMeanScaleMatrixPrecision(μ, γ, G)
    rdist = MvNormalMeanPrecision(μ, γ * G)

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

@testitem "MvNormalMeanScaleMatrixPrecision: Base methods" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalMeanScaleMatrixPrecision{Float32}, MvNormalMeanScaleMatrixPrecision([0.0, 0.0])) ==
          MvNormalMeanScaleMatrixPrecision([0.0f0, 0.0f0], 1.0f0, [1.0 0.0; 0.0 1.0])
    @test convert(MvNormalMeanScaleMatrixPrecision{Float64}, [0.0, 0.0], 2.0) ==
          MvNormalMeanScaleMatrixPrecision([0.0, 0.0], 2.0, [1.0 0.0; 0.0 1.0])
    @test convert(MvNormalMeanScaleMatrixPrecision{Float64}, [0.0, 0.0], 2.0, [1.0 0.0; 0.0 1.0]) ==
          MvNormalMeanScaleMatrixPrecision([0.0, 0.0], 2.0, [1.0 0.0; 0.0 1.0])

    @test length(MvNormalMeanScaleMatrixPrecision([0.0, 0.0])) === 2
    @test length(MvNormalMeanScaleMatrixPrecision([0.0, 0.0, 0.0])) === 3
    @test ndims(MvNormalMeanScaleMatrixPrecision([0.0, 0.0])) === 2
    @test ndims(MvNormalMeanScaleMatrixPrecision([0.0, 0.0, 0.0])) === 3
    @test size(MvNormalMeanScaleMatrixPrecision([0.0, 0.0])) === (2,)
    @test size(MvNormalMeanScaleMatrixPrecision([0.0, 0.0, 0.0])) === (3,)

    μ, γ, G = zeros(2), 2.0, Matrix(I, 2, 2)
    distribution = MvNormalMeanScaleMatrixPrecision(μ, γ, G)

    @test distribution ≈ distribution
    @test convert(MvNormalMeanCovariance, distribution) == MvNormalMeanCovariance(μ, inv(γ) * G)
    @test convert(MvNormalMeanPrecision, distribution) == MvNormalMeanPrecision(μ, γ * G)
    @test convert(MvNormalWeightedMeanPrecision, distribution) == MvNormalWeightedMeanPrecision(γ * μ, γ * G)
end

@testitem "MvNormalMeanScaleMatrixPrecision: vague" begin
    include("./normal_family_setuptests.jl")

    @test_throws MethodError vague(MvNormalMeanScaleMatrixPrecision)

    d1 = vague(MvNormalMeanScaleMatrixPrecision, 2)

    @test typeof(d1) <: MvNormalMeanScaleMatrixPrecision
    @test mean(d1) == zeros(2)
    @test invcov(d1) == Matrix(Diagonal(1e-12 * ones(2)))
    @test ndims(d1) == 2

    d2 = vague(MvNormalMeanScaleMatrixPrecision, 3)

    @test typeof(d2) <: MvNormalMeanScaleMatrixPrecision
    @test mean(d2) == zeros(3)
    @test invcov(d2) == Matrix(Diagonal(1e-12 * ones(3)))
    @test ndims(d2) == 3
end

@testitem "MvNormalMeanScaleMatrixPrecision: prod" begin
    include("./normal_family_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
    @test prod(strategy, MvNormalMeanScaleMatrixPrecision([-1, -1], 2, Matrix(I,2,2)), MvNormalMeanPrecision([1, 1], [2, 4])) ≈
              MvNormalWeightedMeanPrecision([0, 2], [4, 6])

        μ    = [1.0, 2.0, 3.0]
        γ    = 2.0
        G    = Matrix(I, 3, 3)
        dist = MvNormalMeanScaleMatrixPrecision(μ, γ, G)

        @test prod(strategy, dist, dist) ≈
              MvNormalMeanScaleMatrixPrecision([1.0, 2.0, 3.0], 2γ, G)
    end
end

@testitem "MvNormalMeanScaleMatrixPrecision: convert" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalMeanScaleMatrixPrecision, zeros(2), 1.0, Matrix(I,2,2)) ==
          MvNormalMeanScaleMatrixPrecision(zeros(2), 1.0, Matrix(I,2,2))
    @test begin
        m = rand(5)
        c = rand()
    convert(MvNormalMeanScaleMatrixPrecision, m, c, Matrix(I,5,5)) == MvNormalMeanScaleMatrixPrecision(m, c, Matrix(I,5,5))
    end
end

@testitem "MvNormalMeanScaleMatrixPrecision: rand" begin
    include("./normal_family_setuptests.jl")

    rng = MersenneTwister(42)

    for T in (Float32, Float64)
        @testset "Basic functionality" begin
            μ = [1.0, 2.0, 3.0]
            γ = 2.0
            dist = MvNormalMeanScaleMatrixPrecision(convert(Vector{T}, μ), convert(T, γ), Matrix{T}(I,3,3))

            @test typeof(rand(dist)) <: Vector{T}

            samples = rand(rng, dist, 5_000)

            @test isapprox(mean(samples), mean(μ), atol = 0.5)
        end
    end
end