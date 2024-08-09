
@testitem "MvNormalMeanPrecision: Constructor" begin
    include("./normal_family_setuptests.jl")

    @test MvNormalMeanPrecision <: AbstractMvNormal

    @test MvNormalMeanPrecision([1.0, 1.0]) == MvNormalMeanPrecision([1.0, 1.0], [1.0, 1.0])
    @test MvNormalMeanPrecision([1.0, 2.0]) == MvNormalMeanPrecision([1.0, 2.0], [1.0, 1.0])
    @test MvNormalMeanPrecision([1, 2]) == MvNormalMeanPrecision([1.0, 2.0], [1.0, 1.0])
    @test MvNormalMeanPrecision([1.0f0, 2.0f0]) == MvNormalMeanPrecision([1.0f0, 2.0f0], [1.0f0, 1.0f0])

    @test eltype(MvNormalMeanPrecision([1.0, 1.0])) === Float64
    @test eltype(MvNormalMeanPrecision([1.0, 1.0], [1.0, 1.0])) === Float64
    @test eltype(MvNormalMeanPrecision([1, 1])) === Float64
    @test eltype(MvNormalMeanPrecision([1, 1], [1, 1])) === Float64
    @test eltype(MvNormalMeanPrecision([1.0f0, 1.0f0])) === Float32
    @test eltype(MvNormalMeanPrecision([1.0f0, 1.0f0], [1.0f0, 1.0f0])) === Float32

    @test MvNormalMeanPrecision(ones(3), 5I) == MvNormalMeanPrecision(ones(3), Diagonal(5 * ones(3)))
    @test MvNormalMeanPrecision([1, 2, 3, 4], 7.0I) == MvNormalMeanPrecision([1.0, 2.0, 3.0, 4.0], Diagonal(7.0 * ones(4)))
end

@testitem "MvNormalMeanPrecision: distrname" begin
    include("./normal_family_setuptests.jl")

    @test ExponentialFamily.distrname(MvNormalMeanPrecision(zeros(2))) === "MvNormalMeanPrecision"
end

@testitem "MvNormalMeanPrecision: Stats methods" begin
    include("./normal_family_setuptests.jl")

    μ    = [0.2, 3.0, 4.0]
    Λ    = [1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5]
    dist = MvNormalMeanPrecision(μ, Λ)

    @test mean(dist) == μ
    @test mode(dist) == μ
    @test weightedmean(dist) == Λ * μ
    @test invcov(dist) == Λ
    @test precision(dist) == Λ
    @test cov(dist) ≈ inv(Λ)
    @test std(dist) * std(dist)' ≈ inv(Λ)
    @test all(mean_cov(dist) .≈ (μ, inv(Λ)))
    @test all(mean_invcov(dist) .≈ (μ, Λ))
    @test all(mean_precision(dist) .≈ (μ, Λ))
    @test all(weightedmean_cov(dist) .≈ (Λ * μ, inv(Λ)))
    @test all(weightedmean_invcov(dist) .≈ (Λ * μ, Λ))
    @test all(weightedmean_precision(dist) .≈ (Λ * μ, Λ))

    @test length(dist) == 3
    @test entropy(dist) ≈ 3.1517451983126357
    @test pdf(dist, [0.2, 3.0, 4.0]) ≈ 0.19171503573907536
    @test pdf(dist, [0.202, 3.002, 4.002]) ≈ 0.19171258180232315
    @test logpdf(dist, [0.2, 3.0, 4.0]) ≈ -1.6517451983126357
    @test logpdf(dist, [0.202, 3.002, 4.002]) ≈ -1.6517579983126356
end

@testitem "MvNormalMeanPrecision: Base methods" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalMeanPrecision{Float32}, MvNormalMeanPrecision([0.0, 0.0])) ==
          MvNormalMeanPrecision([0.0f0, 0.0f0], [1.0f0, 1.0f0])
    @test convert(MvNormalMeanPrecision{Float64}, [0.0, 0.0], [2 0; 0 3]) ==
          MvNormalMeanPrecision([0.0, 0.0], [2.0 0.0; 0.0 3.0])

    @test length(MvNormalMeanPrecision([0.0, 0.0])) === 2
    @test length(MvNormalMeanPrecision([0.0, 0.0, 0.0])) === 3
    @test ndims(MvNormalMeanPrecision([0.0, 0.0])) === 2
    @test ndims(MvNormalMeanPrecision([0.0, 0.0, 0.0])) === 3
    @test size(MvNormalMeanPrecision([0.0, 0.0])) === (2,)
    @test size(MvNormalMeanPrecision([0.0, 0.0, 0.0])) === (3,)

    distribution = MvNormalMeanPrecision([0.0, 0.0], [2.0 0.0; 0.0 3.0])

    @test distribution ≈ distribution
    @test distribution ≈ convert(MvNormalMeanCovariance, distribution)
    @test distribution ≈ convert(MvNormalWeightedMeanPrecision, distribution)
end

@testitem "MvNormalMeanPrecision: vague" begin
    include("./normal_family_setuptests.jl")

    @test_throws MethodError vague(MvNormalMeanPrecision)

    d1 = vague(MvNormalMeanPrecision, 2)

    @test typeof(d1) <: MvNormalMeanPrecision
    @test mean(d1) == zeros(2)
    @test invcov(d1) == Matrix(Diagonal(1e-12 * ones(2)))
    @test ndims(d1) == 2

    d2 = vague(MvNormalMeanPrecision, 3)

    @test typeof(d2) <: MvNormalMeanPrecision
    @test mean(d2) == zeros(3)
    @test invcov(d2) == Matrix(Diagonal(1e-12 * ones(3)))
    @test ndims(d2) == 3
end

@testitem "MvNormalMeanPrecision: prod" begin
    include("./normal_family_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
        @test prod(strategy, MvNormalMeanPrecision([-1, -1], [2, 2]), MvNormalMeanPrecision([1, 1], [2, 4])) ≈
              MvNormalWeightedMeanPrecision([0, 2], [4, 6])

        μ    = [1.0, 2.0, 3.0]
        Λ    = diagm(1 ./ [1.0, 2.0, 3.0])
        dist = MvNormalMeanPrecision(μ, Λ)

        @test prod(strategy, dist, dist) ≈
              MvNormalWeightedMeanPrecision([2.0, 2.0, 2.0], diagm([2.0, 1.0, 2 / 3]))
    end
end

@testitem "MvNormalMeanPrecision: convert" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalMeanPrecision, zeros(2), Matrix(Diagonal(ones(2)))) ==
          MvNormalMeanPrecision(zeros(2), Matrix(Diagonal(ones(2))))
    @test begin
        m = rand(5)
        c = Matrix(Symmetric(rand(5, 5)))
        convert(MvNormalMeanPrecision, m, c) == MvNormalMeanPrecision(m, c)
    end
end
