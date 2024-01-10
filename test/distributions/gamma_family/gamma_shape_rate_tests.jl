
@testitem "GammaShapeRate: Constructor" begin
    include("./gamma_family_setuptests.jl")

    @test GammaShapeScale <: GammaDistributionsFamily

    @test GammaShapeRate() == GammaShapeRate{Float64}(1.0, 1.0)
    @test GammaShapeRate(1.0) == GammaShapeRate{Float64}(1.0, 1.0)
    @test GammaShapeRate(1.0, 2.0) == GammaShapeRate{Float64}(1.0, 2.0)
    @test GammaShapeRate(1) == GammaShapeRate{Float64}(1.0, 1.0)
    @test GammaShapeRate(1, 2) == GammaShapeRate{Float64}(1.0, 2.0)
    @test GammaShapeRate(1.0, 2) == GammaShapeRate{Float64}(1.0, 2.0)
    @test GammaShapeRate(1, 2.0) == GammaShapeRate{Float64}(1.0, 2.0)
    @test GammaShapeRate(1.0f0) == GammaShapeRate{Float32}(1.0f0, 1.0f0)
    @test GammaShapeRate(1.0f0, 2.0f0) == GammaShapeRate{Float32}(1.0f0, 2.0f0)
    @test GammaShapeRate(1.0f0, 2) == GammaShapeRate{Float32}(1.0f0, 2.0f0)
    @test GammaShapeRate(1.0f0, 2.0) == GammaShapeRate{Float64}(1.0, 2.0)

    @test paramfloattype(GammaShapeRate(1.0, 2.0)) === Float64
    @test paramfloattype(GammaShapeRate(1.0f0, 2.0f0)) === Float32

    @test convert(GammaShapeRate{Float32}, GammaShapeRate()) == GammaShapeRate{Float32}(1.0f0, 1.0f0)
    @test convert(GammaShapeRate{Float64}, GammaShapeRate(1.0, 10.0)) == GammaShapeRate{Float64}(1.0, 10.0)
    @test convert(GammaShapeRate{Float64}, GammaShapeRate(1.0, 0.1)) == GammaShapeRate{Float64}(1.0, 0.1)
    @test convert(GammaShapeRate{Float64}, 1, 1) == GammaShapeRate{Float64}(1.0, 1.0)
    @test convert(GammaShapeRate{Float64}, 1, 10) == GammaShapeRate{Float64}(1.0, 10.0)
    @test convert(GammaShapeRate{Float64}, 1.0, 0.1) == GammaShapeRate{Float64}(1.0, 0.1)

    @test convert(GammaShapeRate, GammaShapeRate(2.0, 2.0)) == GammaShapeRate{Float64}(2.0, 2.0)
    @test convert(GammaShapeScale, GammaShapeRate(2.0, 2.0)) == GammaShapeScale{Float64}(2.0, 1.0 / 2.0)

    @test convert(GammaShapeRate, GammaShapeScale(2.0, 2.0)) == GammaShapeRate{Float64}(2.0, 1.0 / 2.0)
    @test convert(GammaShapeScale, GammaShapeScale(2.0, 2.0)) == GammaShapeScale{Float64}(2.0, 2.0)
end

@testitem "GammaShapeRate: vague" begin
    include("./gamma_family_setuptests.jl")

    @test vague(GammaShapeRate) == GammaShapeRate(1.0, 1e-12)
end

@testitem "GammaShapeRate: stats methods" begin
    include("./gamma_family_setuptests.jl")

    dist1 = GammaShapeRate(1.0, 1.0)

    @test mean(dist1) === 1.0
    @test var(dist1) === 1.0
    @test cov(dist1) === 1.0
    @test shape(dist1) === 1.0
    @test scale(dist1) === 1.0
    @test rate(dist1) === 1.0
    @test entropy(dist1) ≈ 1.0
    @test pdf(dist1, 1.0) ≈ 0.36787944117144233
    @test logpdf(dist1, 1.0) ≈ -1.0

    dist2 = GammaShapeRate(1.0, 2.0)

    @test mean(dist2) === inv(2.0)
    @test var(dist2) === inv(4.0)
    @test cov(dist2) === inv(4.0)
    @test shape(dist2) === 1.0
    @test scale(dist2) === inv(2.0)
    @test rate(dist2) === 2.0
    @test entropy(dist2) ≈ 0.3068528194400547
    @test pdf(dist2, 1.0) ≈ 0.2706705664732254
    @test logpdf(dist2, 1.0) ≈ -1.3068528194400546

    dist3 = GammaShapeRate(2.0, 2.0)

    @test mean(dist3) === 1.0
    @test var(dist3) === inv(2.0)
    @test cov(dist3) === inv(2.0)
    @test shape(dist3) === 2.0
    @test scale(dist3) === inv(2.0)
    @test rate(dist3) === 2.0
    @test entropy(dist3) ≈ 0.8840684843415857
    @test pdf(dist3, 1.0) ≈ 0.5413411329464508
    @test logpdf(dist3, 1.0) ≈ -0.6137056388801094

    # see https://github.com/ReactiveBayes/ReactiveMP.jl/issues/314
    dist = GammaShapeRate(257.37489915581654, 3.0)
    @test pdf(dist, 86.2027941354432) == 0.07400338986721687
end

@testitem "GammaShapeRate: prod" begin
    include("./gamma_family_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
        @test prod(strategy, GammaShapeScale(1, 1), GammaShapeScale(1, 1)) == GammaShapeScale(1, 1 / 2)
        @test prod(strategy, GammaShapeScale(1, 2), GammaShapeScale(1, 1)) == GammaShapeScale(1, 2 / 3)
        @test prod(strategy, GammaShapeScale(1, 2), GammaShapeScale(1, 2)) == GammaShapeScale(1, 1)
        @test prod(strategy, GammaShapeScale(2, 2), GammaShapeScale(1, 2)) == GammaShapeScale(2, 1)
        @test prod(strategy, GammaShapeScale(2, 2), GammaShapeScale(2, 2)) == GammaShapeScale(3, 1)
        @test prod(strategy, GammaShapeScale(1, 1), GammaShapeRate(1, 1)) == GammaShapeScale(1, 1 / 2)
        @test prod(strategy, GammaShapeScale(1, 2), GammaShapeRate(1, 1)) == GammaShapeScale(1, 2 / 3)
        @test prod(strategy, GammaShapeScale(1, 2), GammaShapeRate(1, 2)) == GammaShapeScale(1, 2 / 5)
        @test prod(strategy, GammaShapeScale(2, 2), GammaShapeRate(1, 2)) == GammaShapeScale(2, 2 / 5)
        @test prod(strategy, GammaShapeScale(2, 2), GammaShapeRate(2, 2)) == GammaShapeScale(3, 2 / 5)
    end
end
