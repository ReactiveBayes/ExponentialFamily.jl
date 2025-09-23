using LinearAlgebra
using StaticArrays

SUITE["mvnormal_weighted_mean_precision"] = BenchmarkGroup(
    ["mvnormal_weighted_mean_precision", "normal_family", "distribution"],
    "prod" => BenchmarkGroup(["prod", "multiplication"])
)

# Helpers ==========================
spd_matrix(T::Type, d::Integer) = begin
    A = rand(T, d, d)
    # Make symmetric positive definite and well-conditioned enough for Float16
    Σ = A' * A
    Matrix{T}(Symmetric(Σ))
end

dense_dist(::Type{T}, d::Integer) where {T <: Real} =
    MvNormalWeightedMeanPrecision(rand(T, d), spd_matrix(T, d))

diag_dist(::Type{T}, d::Integer) where {T <: Real} = begin
    μ = rand(T, d)
    σ = abs.(rand(T, d)) .+ one(T)
    MvNormalMeanCovariance(μ, Diagonal(σ))
end

static_dist(::Type{T}, ::Val{D}) where {T <: Real, D} = begin
    μ = SVector{D, T}(rand(T, D))
    A = SMatrix{D, D, T}(rand(T, D, D))
    Σ = A' * A
    MvNormalMeanCovariance(μ, Σ)
end

# prod (PreserveType) ==============
let dims_dense = (10, 50, 100)
    # Dense × Dense (BLAS path for Float32/64)
    for d in dims_dense
        for (TL, TR) in ((Float64, Float64), (Float32, Float32), (Float32, Float64))
            left = dense_dist(TL, d)
            right = dense_dist(TR, d)
            SUITE["mvnormal_weighted_mean_precision"]["prod"]["PreserveType"]["Dense×Dense"]["d=$d"]["$(TL)×$(TR)"] =
                @benchmarkable prod(PreserveTypeProd(Distribution), $left, $right)
        end
        # Mixed lower precision (generic path)
        let TL = Float16, TR = Float64
            left = dense_dist(TL, d)
            right = dense_dist(TR, d)
            SUITE["mvnormal_weighted_mean_precision"]["prod"]["PreserveType"]["Dense×Dense"]["d=$d"]["Float16×Float64"] =
                @benchmarkable prod(PreserveTypeProd(Distribution), $left, $right)
        end
    end

    # Dense × Diagonal
    for d in dims_dense
        left = dense_dist(Float64, d)
        right = diag_dist(Float64, d)
        SUITE["mvnormal_weighted_mean_precision"]["prod"]["PreserveType"]["Dense×Diag"]["d=$d"]["Float64×Float64"] =
            @benchmarkable prod(PreserveTypeProd(Distribution), $left, $right)

        left = dense_dist(Float32, d)
        right = diag_dist(Float64, d)
        SUITE["mvnormal_weighted_mean_precision"]["prod"]["PreserveType"]["Dense×Diag"]["d=$d"]["Float32×Float64"] =
            @benchmarkable prod(PreserveTypeProd(Distribution), $left, $right)
    end

    # Diagonal × Diagonal
    for d in (10, 50, 100)
        left = diag_dist(Float64, d)
        right = diag_dist(Float64, d)
        SUITE["mvnormal_weighted_mean_precision"]["prod"]["PreserveType"]["Diag×Diag"]["d=$d"]["Float64×Float64"] =
            @benchmarkable prod(PreserveTypeProd(Distribution), $left, $right)
    end

    # StaticArrays (small d only)
    for D in (3, 6)
        for TL in (Float64, Float32, Float16)
            for TR in (Float64, Float32, Float16)
                # Static × Static
                left = static_dist(TL, Val(D))
                right = static_dist(TR, Val(D))
                SUITE["mvnormal_weighted_mean_precision"]["prod"]["PreserveType"]["SArray×SArray"]["d=$D"]["$(TL)×$(TR)"] =
                    @benchmarkable prod(PreserveTypeProd(Distribution), $left, $right)

                # Static × Dense
                left = static_dist(TL, Val(D))
                right = dense_dist(TR, D)
                SUITE["mvnormal_weighted_mean_precision"]["prod"]["PreserveType"]["SArray×Dense"]["d=$D"]["$(TL)×$(TR)"] =
                    @benchmarkable prod(PreserveTypeProd(Distribution), $left, $right)

                # Static × Diagonal
                left = static_dist(TL, Val(D))
                right = diag_dist(TR, D)
                SUITE["mvnormal_weighted_mean_precision"]["prod"]["PreserveType"]["SArray×Diag"]["d=$D"]["$(TL)×$(TR)"] =
                    @benchmarkable prod(PreserveTypeProd(Distribution), $left, $right)
            end
        end
    end
end
