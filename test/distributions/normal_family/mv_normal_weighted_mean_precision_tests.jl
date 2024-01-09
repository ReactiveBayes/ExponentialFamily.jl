
@testitem "MvNormalWeightedMeanPrecision: Constructor" begin
    include("./normal_family_setuptests.jl")

    @test MvNormalWeightedMeanPrecision <: AbstractMvNormal

    @test MvNormalWeightedMeanPrecision([1.0, 1.0]) == MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0, 1.0])
    @test MvNormalWeightedMeanPrecision([1.0, 2.0]) == MvNormalWeightedMeanPrecision([1.0, 2.0], [1.0, 1.0])
    @test MvNormalWeightedMeanPrecision([1, 2]) == MvNormalWeightedMeanPrecision([1.0, 2.0], [1.0, 1.0])
    @test MvNormalWeightedMeanPrecision([1.0f0, 2.0f0]) ==
          MvNormalWeightedMeanPrecision([1.0f0, 2.0f0], [1.0f0, 1.0f0])

    @test eltype(MvNormalWeightedMeanPrecision([1.0, 1.0])) === Float64
    @test eltype(MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0, 1.0])) === Float64
    @test eltype(MvNormalWeightedMeanPrecision([1, 1])) === Float64
    @test eltype(MvNormalWeightedMeanPrecision([1, 1], [1, 1])) === Float64
    @test eltype(MvNormalWeightedMeanPrecision([1.0f0, 1.0f0])) === Float32
    @test eltype(MvNormalWeightedMeanPrecision([1.0f0, 1.0f0], [1.0f0, 1.0f0])) === Float32
end

@testitem "MvNormalWeightedMeanPrecision: distrname" begin
    include("./normal_family_setuptests.jl")

    @test ExponentialFamily.distrname(MvNormalWeightedMeanPrecision(zeros(2))) === "MvNormalWeightedMeanPrecision"
end

@testitem "MvNormalWeightedMeanPrecision: Stats methods" begin
    include("./normal_family_setuptests.jl")

    xi   = [-0.2, 5.34, 14.02]
    Λ    = [1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5]
    dist = MvNormalWeightedMeanPrecision(xi, Λ)

    @test mean(dist) ≈ inv(Λ) * xi
    @test mode(dist) ≈ inv(Λ) * xi
    @test weightedmean(dist) == xi
    @test invcov(dist) == Λ
    @test precision(dist) == Λ
    @test cov(dist) ≈ inv(Λ)
    @test std(dist) * std(dist)' ≈ inv(Λ)
    @test all(mean_cov(dist) .≈ (inv(Λ) * xi, inv(Λ)))
    @test all(mean_invcov(dist) .≈ (inv(Λ) * xi, Λ))
    @test all(mean_precision(dist) .≈ (inv(Λ) * xi, Λ))
    @test all(weightedmean_cov(dist) .≈ (xi, inv(Λ)))
    @test all(weightedmean_invcov(dist) .≈ (xi, Λ))
    @test all(weightedmean_precision(dist) .≈ (xi, Λ))

    @test length(dist) == 3
    @test entropy(dist) ≈ 3.1517451983126357
    @test pdf(dist, [0.2, 3.0, 4.0]) ≈ 0.19171503573907536
    @test pdf(dist, [0.202, 3.002, 4.002]) ≈ 0.19171258180232315
    @test logpdf(dist, [0.2, 3.0, 4.0]) ≈ -1.6517451983126357
    @test logpdf(dist, [0.202, 3.002, 4.002]) ≈ -1.6517579983126356
end

@testitem "MvNormalWeightedMeanPrecision: Base methods" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalWeightedMeanPrecision{Float32}, MvNormalWeightedMeanPrecision([0.0, 0.0])) ==
          MvNormalWeightedMeanPrecision([0.0f0, 0.0f0], [1.0f0, 1.0f0])
    @test convert(MvNormalWeightedMeanPrecision{Float64}, [0.0, 0.0], [2 0; 0 3]) ==
          MvNormalWeightedMeanPrecision([0.0, 0.0], [2.0 0.0; 0.0 3.0])

    @test length(MvNormalWeightedMeanPrecision([0.0, 0.0])) === 2
    @test length(MvNormalWeightedMeanPrecision([0.0, 0.0, 0.0])) === 3
    @test ndims(MvNormalWeightedMeanPrecision([0.0, 0.0])) === 2
    @test ndims(MvNormalWeightedMeanPrecision([0.0, 0.0, 0.0])) === 3
    @test size(MvNormalWeightedMeanPrecision([0.0, 0.0])) === (2,)
    @test size(MvNormalWeightedMeanPrecision([0.0, 0.0, 0.0])) === (3,)

    distribution = MvNormalWeightedMeanPrecision([0.0, 0.0], [2.0 0.0; 0.0 3.0])

    @test distribution ≈ distribution
    @test distribution ≈ convert(MvNormalMeanCovariance, distribution)
    @test distribution ≈ convert(MvNormalMeanPrecision, distribution)
end

@testitem "MvNormalWeightedMeanPrecision: vague" begin
    include("./normal_family_setuptests.jl")

    @test_throws MethodError vague(MvNormalWeightedMeanPrecision)

    d1 = vague(MvNormalWeightedMeanPrecision, 2)

    @test typeof(d1) <: MvNormalWeightedMeanPrecision
    @test mean(d1) == zeros(2)
    @test invcov(d1) == Matrix(Diagonal(1e-12 * ones(2)))
    @test ndims(d1) == 2

    d2 = vague(MvNormalWeightedMeanPrecision, 3)

    @test typeof(d2) <: MvNormalWeightedMeanPrecision
    @test mean(d2) == zeros(3)
    @test invcov(d2) == Matrix(Diagonal(1e-12 * ones(3)))
    @test ndims(d2) == 3
end

@testitem "MvNormalWeightedMeanPrecision: prod" begin
    include("./normal_family_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
        @test prod(
            strategy,
            MvNormalWeightedMeanPrecision([-1, -1], [2, 2]),
            MvNormalWeightedMeanPrecision([1, 1], [2, 4])
        ) ≈ MvNormalWeightedMeanPrecision([0, 0], [4, 6])

        xi   = [0.2, 3.0, 4.0]
        Λ    = [1.5 -0.1 0.1; -0.1 1.8 0.0; 0.1 0.0 3.5]
        dist = MvNormalWeightedMeanPrecision(xi, Λ)

        @test prod(strategy, dist, dist) ≈
              MvNormalWeightedMeanPrecision([0.40, 6.00, 8.00], [3.00 -0.20 0.20; -0.20 3.60 0.00; 0.20 0.00 7.00])
    end
end

@testitem "MvNormalWeightedMeanPrecision: convert" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalWeightedMeanPrecision, zeros(2), Matrix(Diagonal(ones(2)))) ==
          MvNormalWeightedMeanPrecision(zeros(2), Matrix(Diagonal(ones(2))))
    @test begin
        m = rand(5)
        c = Matrix(Symmetric(rand(5, 5)))
        convert(MvNormalWeightedMeanPrecision, m, c) == MvNormalWeightedMeanPrecision(m, c)
    end
end
