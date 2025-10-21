
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

    @test MvNormalWeightedMeanPrecision(ones(3), 5I) == MvNormalWeightedMeanPrecision(ones(3), Diagonal(5 * ones(3)))
    @test MvNormalWeightedMeanPrecision([1, 2, 3, 4], 7.0I) == MvNormalWeightedMeanPrecision([1.0, 2.0, 3.0, 4.0], Diagonal(7.0 * ones(4)))
end

@testitem "MvNormalWeightedMeanPrecision: prod" begin
    include("./normal_family_setuptests.jl")
    using LinearAlgebra, StaticArrays

    mk_dense(T) = MvNormalWeightedMeanPrecision(T[1, 4, 3], [2 1 1; 1 3 1; 1 1 4])
    mk_diag(T)  = MvNormalWeightedMeanPrecision(T[2, 2, 1], Diagonal(T[2, 3, 4]))
    mk_sarr(T)  = MvNormalWeightedMeanPrecision(
    SVector{3, T}(1, 2, 3),
    @SMatrix [T(2) 0 0; 0 T(3) 0; 0 0 T(4)]
)
    mk_mixed(T) = MvNormalWeightedMeanPrecision(SVector{3, T}(1, 2, 3), T[2 1 1; 1 3 1; 1 1 4])

    cases = [
        ("Dense64×Dense64", mk_dense(Float64), mk_dense(Float64)),
        ("Dense32×Dense32", mk_dense(Float32), mk_dense(Float32)),
        ("Dense32×Dense64", mk_dense(Float32), mk_dense(Float64)),
        ("Dense16×Dense64", mk_dense(Float16), mk_dense(Float64)),
        ("Dense64×Diag64", mk_dense(Float64), mk_diag(Float64)),
        ("Diag64×Diag64", mk_diag(Float64), mk_diag(Float64)),
        ("SArray64×SArray64", mk_sarr(Float64), mk_sarr(Float64)),
        ("SArray16×Dense64", mk_sarr(Float16), mk_dense(Float64)),
        ("Dense64×Mixed64", mk_dense(Float64), mk_mixed(Float64)),
        ("Mixed64×Dense64", mk_mixed(Float64), mk_dense(Float64)),
        ("Mixed64×Mixed64", mk_mixed(Float64), mk_mixed(Float64))
    ]

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
        @test prod(
            strategy,
            MvNormalWeightedMeanPrecision([-1, -1], [2, 2]),
            MvNormalWeightedMeanPrecision([1, 1], [2, 4])
        ) ≈ MvNormalWeightedMeanPrecision([0, 0], [4, 6])
        for (label, left, right) in cases
            res = prod(strategy, left, right)
            xiL, WL = weightedmean_precision(left)
            xiR, WR = weightedmean_precision(right)
            expected = MvNormalWeightedMeanPrecision(xiL + xiR, WL + WR)

            @test res ≈ expected
            @test prod(strategy, right, left) ≈ expected
        end
    end

    left  = mk_dense(Float64)  # d=3
    right = MvNormalWeightedMeanPrecision(Float64[1, 2], diagm(Float64[2, 3])) # d=2
    @test_throws DimensionMismatch prod(PreserveTypeProd(Distribution), left, right)
end

@testitem "MvNormalWeightedMeanPrecision: distrname" begin
    include("./normal_family_setuptests.jl")

    @test ExponentialFamily.distrname(MvNormalWeightedMeanPrecision(zeros(2))) === "MvNormalWeightedMeanPrecision"
end

@testitem "MvNormalWeightedMeanPrecision: Stats methods" begin
    include("./normal_family_setuptests.jl")

    xi   = [-0.2, 5.34, 14.02]
    Λ   = [1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5]
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
