
@testitem "GammaShapeScale: Constructor" begin
    include("./gamma_family_setuptests.jl")

    @test GammaShapeScale <: GammaDistributionsFamily

    @test GammaShapeScale() == GammaShapeScale{Float64}(1.0, 1.0)
    @test GammaShapeScale(1.0) == GammaShapeScale{Float64}(1.0, 1.0)
    @test GammaShapeScale(1.0, 2.0) == GammaShapeScale{Float64}(1.0, 2.0)
    @test GammaShapeScale(1) == GammaShapeScale{Float64}(1.0, 1.0)
    @test GammaShapeScale(1, 2) == GammaShapeScale{Float64}(1.0, 2.0)
    @test GammaShapeScale(1.0, 2) == GammaShapeScale{Float64}(1.0, 2.0)
    @test GammaShapeScale(1, 2.0) == GammaShapeScale{Float64}(1.0, 2.0)
    @test GammaShapeScale(1.0f0) == GammaShapeScale{Float32}(1.0f0, 1.0f0)
    @test GammaShapeScale(1.0f0, 2.0f0) == GammaShapeScale{Float32}(1.0f0, 2.0f0)
    @test GammaShapeScale(1.0f0, 2) == GammaShapeScale{Float32}(1.0f0, 2.0f0)
    @test GammaShapeScale(1.0f0, 2.0) == GammaShapeScale{Float64}(1.0, 2.0)

    @test paramfloattype(GammaShapeScale(1.0, 2.0)) === Float64
    @test paramfloattype(GammaShapeScale(1.0f0, 2.0f0)) === Float32

    @test convert(GammaShapeScale{Float32}, GammaShapeScale()) == GammaShapeScale{Float32}(1.0f0, 1.0f0)
    @test convert(GammaShapeScale{Float64}, GammaShapeScale(1.0, 10.0)) == GammaShapeScale{Float64}(1.0, 10.0)
    @test convert(GammaShapeScale{Float64}, GammaShapeScale(1.0, 0.1)) == GammaShapeScale{Float64}(1.0, 0.1)
    @test convert(GammaShapeScale{Float64}, 1, 1) == GammaShapeScale{Float64}(1.0, 1.0)
    @test convert(GammaShapeScale{Float64}, 1, 10) == GammaShapeScale{Float64}(1.0, 10.0)
    @test convert(GammaShapeScale{Float64}, 1.0, 0.1) == GammaShapeScale{Float64}(1.0, 0.1)

    @test convert(GammaShapeRate, GammaShapeScale(2.0, 2.0)) == GammaShapeRate{Float64}(2.0, 1.0 / 2.0)
    @test convert(GammaShapeScale, GammaShapeScale(2.0, 2.0)) == GammaShapeScale{Float64}(2.0, 2.0)
end

@testitem "GammaShapeScale: vague" begin
    include("./gamma_family_setuptests.jl")

    @test vague(GammaShapeScale) == GammaShapeScale(1.0, 1e12)
end

@testitem "GammaShapeScale: stats methods" begin
    include("./gamma_family_setuptests.jl")

    dist1 = GammaShapeScale(1.0, 1.0)

    @test mean(dist1) === 1.0
    @test var(dist1) === 1.0
    @test cov(dist1) === 1.0
    @test shape(dist1) === 1.0
    @test scale(dist1) === 1.0
    @test rate(dist1) === 1.0
    @test entropy(dist1) ≈ 1.0

    dist2 = GammaShapeScale(1.0, 2.0)

    @test mean(dist2) === 2.0
    @test var(dist2) === 4.0
    @test cov(dist2) === 4.0
    @test shape(dist2) === 1.0
    @test scale(dist2) === 2.0
    @test rate(dist2) === inv(2.0)
    @test entropy(dist2) ≈ 1.6931471805599454

    dist3 = GammaShapeScale(2.0, 2.0)

    @test mean(dist3) === 4.0
    @test var(dist3) === 8.0
    @test cov(dist3) === 8.0
    @test shape(dist3) === 2.0
    @test scale(dist3) === 2.0
    @test rate(dist3) === inv(2.0)
    @test entropy(dist3) ≈ 2.2703628454614764
end

@testitem "GammaShapeScale: prod" begin
    include("./gamma_family_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
        @test prod(strategy, GammaShapeRate(1, 1), GammaShapeRate(1, 1)) == GammaShapeRate(1, 2)
        @test prod(strategy, GammaShapeRate(1, 2), GammaShapeRate(1, 1)) == GammaShapeRate(1, 3)
        @test prod(strategy, GammaShapeRate(1, 2), GammaShapeRate(1, 2)) == GammaShapeRate(1, 4)
        @test prod(strategy, GammaShapeRate(2, 2), GammaShapeRate(1, 2)) == GammaShapeRate(2, 4)
        @test prod(strategy, GammaShapeRate(2, 2), GammaShapeRate(2, 2)) == GammaShapeRate(3, 4)
        @test prod(strategy, GammaShapeRate(1, 1), GammaShapeScale(1, 1)) == GammaShapeRate(1, 2)
        @test prod(strategy, GammaShapeRate(1, 2), GammaShapeScale(1, 1)) == GammaShapeRate(1, 3)
        @test prod(strategy, GammaShapeRate(1, 2), GammaShapeScale(1, 2)) == GammaShapeRate(1, 5 / 2)
        @test prod(strategy, GammaShapeRate(2, 2), GammaShapeScale(1, 2)) == GammaShapeRate(2, 5 / 2)
        @test prod(strategy, GammaShapeRate(2, 2), GammaShapeScale(2, 2)) == GammaShapeRate(3, 5 / 2)
    end
end