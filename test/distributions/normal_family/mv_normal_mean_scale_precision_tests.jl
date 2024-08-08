
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
    @test MvNormalMeanScalePrecision([1, 2, 3, 4], 7.0) == MvNormalMeanPrecision([1.0, 2.0, 3.0, 4.0], 7.0)
end

@testitem "MvNormalMeanScalePrecision: distrname" begin
    include("./normal_family_setuptests.jl")

    @test ExponentialFamily.distrname(MvNormalMeanScalePrecision(zeros(2))) === "MvNormalMeanScalePrecision"
end

@testitem "MvNormalMeanScalePrecision: Stats methods" begin
    include("./normal_family_setuptests.jl")

    μ    = [0.2, 3.0, 4.0]
    γ    = 2.0
    dist = MvNormalMeanScalePrecision(μ, γ)
    rdist = MvNormalMeanPrecision(μ, γ * ones(length(μ)))

    @test mean(dist) == μ
    @test mode(dist) == μ
    @test weightedmean(dist) == γ * μ
    @test invcov(dist) == γ
    @test precision(dist) == γ
    @test cov(dist) ≈ inv(Λ)
    @test std(dist) * std(dist)' ≈ inv(γ)
    @test all(mean_cov(dist) .≈ (μ, inv(γ)))
    @test all(mean_invcov(dist) .≈ (μ, γ))
    @test all(mean_precision(dist) .≈ (μ, γ))
    @test all(weightedmean_cov(dist) .≈ (Λ * μ, inv(γ)))
    @test all(weightedmean_invcov(dist) .≈ (γ * μ, γ))
    @test all(weightedmean_precision(dist) .≈ (γ * μ, γ))

    @test length(dist) == 3
    @test entropy(dist) ≈ entropy(rdist)
    @test pdf(dist, [0.2, 3.0, 4.0]) ≈ pdf(rdist, [0.2, 3.0, 4.0])
    @test pdf(dist, [0.202, 3.002, 4.002]) ≈ pdf(rdist, [0.2, 3.0, 4.0])
    @test logpdf(dist, [0.2, 3.0, 4.0]) ≈ pdf(rdist, [0.2, 3.0, 4.0])
    @test logpdf(dist, [0.202, 3.002, 4.002]) ≈ pdf(rdist, [0.2, 3.0, 4.0])
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
    
    distribution = MvNormalMeanScalePrecision([0.0, 0.0], [2.0 0.0; 0.0 3.0])

    @test distribution ≈ distribution
    @test distribution ≈ convert(MvNormalMeanCovariance, distribution)
    @test distribution ≈ convert(MvNormalMeanPrecision, distribution)
    @test distribution ≈ convert(MvNormalWeightedMeanPrecision, distribution)
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
        dist = MvNormalMeanPrecision(μ, Λ)

        @test prod(strategy, dist, dist) ≈
            MvNormalMeanScalePrecision([4.0, 8.0, 12.0], 2γ)
    end
end

@testitem "MvNormalMeanScalePrecision: convert" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalMeanScalePrecision, zeros(2), 1.0) ==
        MvNormalMeanScalePrecision(zeros(2), 1.0)
    @test begin
        m = rand(5)
        c = rand()
        convert(MvNormalMeanScalePrecision, m, c) == MvNormalMeanPrecision(m, c)
    end
end
