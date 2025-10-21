
@testitem "MvNormalMeanCovariance: Constructor" begin
    include("./normal_family_setuptests.jl")

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

    @test MvNormalMeanCovariance(ones(3), 5I) == MvNormalMeanCovariance(ones(3), Diagonal(5 * ones(3)))
    @test MvNormalMeanCovariance([1, 2, 3, 4], 7.0I) == MvNormalMeanCovariance([1.0, 2.0, 3.0, 4.0], Diagonal(7.0 * ones(4)))
end

@testitem "MvNormalMeanCovariance: distrname" begin
    include("./normal_family_setuptests.jl")

    @test ExponentialFamily.distrname(MvNormalMeanCovariance(zeros(2))) === "MvNormalMeanCovariance"
end

@testitem "MvNormalMeanCovariance: Stats methods" begin
    include("./normal_family_setuptests.jl")

    μ = [0.2, 3.0, 4.0]
    Σ = [1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5]
    dist = MvNormalMeanCovariance(μ, Σ)

    @test mean(dist) == μ
    @test mode(dist) == μ
    @test weightedmean(dist) ≈ inv(Σ) * μ
    @test invcov(dist) ≈ inv(Σ)
    @test precision(dist) ≈ inv(Σ)
    @test cov(dist) == Σ
    @test std(dist) * std(dist)' ≈ Σ
    @test all(mean_cov(dist) .≈ (μ, Σ))
    @test all(mean_invcov(dist) .≈ (μ, inv(Σ)))
    @test all(mean_precision(dist) .≈ (μ, inv(Σ)))
    @test all(weightedmean_cov(dist) .≈ (inv(Σ) * μ, Σ))
    @test all(weightedmean_invcov(dist) .≈ (inv(Σ) * μ, inv(Σ)))
    @test all(weightedmean_precision(dist) .≈ (inv(Σ) * μ, inv(Σ)))

    @test length(dist) == 3
    @test entropy(dist) ≈ 5.361886000915401
    @test pdf(dist, [0.2, 3.0, 4.0]) ≈ 0.021028302702542
    @test pdf(dist, [0.202, 3.002, 4.002]) ≈ 0.021028229679079503
    @test logpdf(dist, [0.2, 3.0, 4.0]) ≈ -3.8618860009154012
    @test logpdf(dist, [0.202, 3.002, 4.002]) ≈ -3.861889473548943
end

@testitem "MvNormalMeanCovariance: Base methods" begin
    include("./normal_family_setuptests.jl")

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

    distribution = MvNormalMeanCovariance([0.0, 0.0], [2.0 0.0; 0.0 3.0])

    @test distribution ≈ distribution
    @test distribution ≈ convert(MvNormalMeanPrecision, distribution)
    @test distribution ≈ convert(MvNormalWeightedMeanPrecision, distribution)
end

@testitem "MvNormalMeanCovariance: vague" begin
    include("./normal_family_setuptests.jl")

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

@testitem "MvNormalMeanCovariance: prod" begin
    include("./normal_family_setuptests.jl")
    using LinearAlgebra, StaticArrays

    mk_dense(T) = MvNormalMeanCovariance(T[1, 4, 3], [2 1 1; 1 3 1; 1 1 4])
    mk_diag(T)  = MvNormalMeanCovariance(T[2, 2, 1], Diagonal(T[2, 3, 4]))
    mk_sarr(T)  = MvNormalMeanCovariance(
    SVector{3, T}(1, 2, 3),
    @SMatrix [T(2) 0 0; 0 T(3) 0; 0 0 T(4)]
)
    mk_mixed(T) = MvNormalMeanCovariance(SVector{3, T}(1, 2, 3), T[2 1 1; 1 3 1; 1 1 4])

    cases = [
        ("Dense64×Dense64", mk_dense(Float64), mk_dense(Float64)),  # BLAS path
        ("Dense32×Dense32", mk_dense(Float32), mk_dense(Float32)),  # BLAS path
        ("Dense32×Dense64", mk_dense(Float32), mk_dense(Float64)),  # generic
        ("Dense16×Dense64", mk_dense(Float16), mk_dense(Float64)),  # generic
        ("Dense64×Diag64", mk_dense(Float64), mk_diag(Float64)),   # generic
        ("Diag64×Diag64", mk_diag(Float64), mk_diag(Float64)),   # generic
        ("SArray64×SArray64", mk_sarr(Float64), mk_sarr(Float64)),   # generic
        ("SArray16×Dense64", mk_sarr(Float16), mk_dense(Float64)),  # generic
        ("Dense64×Mixed64", mk_dense(Float64), mk_mixed(Float64)),  # generic
        ("Mixed64×Dense64", mk_mixed(Float64), mk_dense(Float64)),  # generic
        ("Mixed64×Mixed64", mk_mixed(Float64), mk_mixed(Float64))  # generic
    ]

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
        @test prod(strategy, MvNormalMeanCovariance([-1, -1], [2, 2]), MvNormalMeanCovariance([1, 1], [2, 4])) ≈
              MvNormalWeightedMeanPrecision([0, -1 / 4], [1, 3 / 4])
        for (label, left, right) in cases
            res = prod(strategy, left, right)
            xiL, WL = weightedmean_precision(left)
            xiR, WR = weightedmean_precision(right)
            expected = MvNormalWeightedMeanPrecision(xiL + xiR, WL + WR)

            @test res ≈ expected
            # Test that swapping left and right produces the same result
            @test prod(strategy, right, left) ≈ expected
        end
    end

    # Dimension mismatch should error
    left  = mk_dense(Float64)  # d=3
    right = MvNormalMeanCovariance(Float64[1, 2], diagm(Float64[2, 3])) # d=2
    @test_throws DimensionMismatch prod(PreserveTypeProd(Distribution), left, right)
end

@testitem "MvNormalMeanCovariance: convert" begin
    include("./normal_family_setuptests.jl")

    @test convert(MvNormalMeanCovariance, zeros(2), Matrix(Diagonal(ones(2)))) ==
          MvNormalMeanCovariance(zeros(2), Matrix(Diagonal(ones(2))))
    @test begin
        m = rand(5)
        c = Matrix(Symmetric(rand(5, 5)))
        convert(MvNormalMeanCovariance, m, c) == MvNormalMeanCovariance(m, c)
    end
end
