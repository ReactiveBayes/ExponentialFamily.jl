
@testitem "NormalWeightedMeanPrecision: Constructor" begin
    include("./normal_family_setuptests.jl")

    @test NormalWeightedMeanPrecision <: NormalDistributionsFamily
    @test NormalWeightedMeanPrecision <: UnivariateNormalDistributionsFamily

    @test NormalWeightedMeanPrecision() == NormalWeightedMeanPrecision{Float64}(0.0, 1.0)
    @test NormalWeightedMeanPrecision(1.0) == NormalWeightedMeanPrecision{Float64}(1.0, 1.0)
    @test NormalWeightedMeanPrecision(1.0, 2.0) == NormalWeightedMeanPrecision{Float64}(1.0, 2.0)
    @test NormalWeightedMeanPrecision(1) == NormalWeightedMeanPrecision{Float64}(1.0, 1.0)
    @test NormalWeightedMeanPrecision(1, 2) == NormalWeightedMeanPrecision{Float64}(1.0, 2.0)
    @test NormalWeightedMeanPrecision(1.0, 2) == NormalWeightedMeanPrecision{Float64}(1.0, 2.0)
    @test NormalWeightedMeanPrecision(1, 2.0) == NormalWeightedMeanPrecision{Float64}(1.0, 2.0)
    @test NormalWeightedMeanPrecision(1.0f0) == NormalWeightedMeanPrecision{Float32}(1.0f0, 1.0f0)
    @test NormalWeightedMeanPrecision(1.0f0, 2.0f0) == NormalWeightedMeanPrecision{Float32}(1.0f0, 2.0f0)
    @test NormalWeightedMeanPrecision(1.0f0, 2.0) == NormalWeightedMeanPrecision{Float64}(1.0, 2.0)

    @test eltype(NormalWeightedMeanPrecision()) === Float64
    @test eltype(NormalWeightedMeanPrecision(0.0)) === Float64
    @test eltype(NormalWeightedMeanPrecision(0.0, 1.0)) === Float64
    @test eltype(NormalWeightedMeanPrecision(0)) === Float64
    @test eltype(NormalWeightedMeanPrecision(0, 1)) === Float64
    @test eltype(NormalWeightedMeanPrecision(0.0, 1)) === Float64
    @test eltype(NormalWeightedMeanPrecision(0, 1.0)) === Float64
    @test eltype(NormalWeightedMeanPrecision(0.0f0)) === Float32
    @test eltype(NormalWeightedMeanPrecision(0.0f0, 1.0f0)) === Float32
    @test eltype(NormalWeightedMeanPrecision(0.0f0, 1.0)) === Float64

    @test NormalWeightedMeanPrecision(3, 5I) == NormalWeightedMeanPrecision(3, 5)
    @test NormalWeightedMeanPrecision(2, 7.0I) == NormalWeightedMeanPrecision(2.0, 7.0)
end

@testitem "NormalWeightedMeanPrecision: Stats methods" begin
    include("./normal_family_setuptests.jl")

    dist1 = NormalWeightedMeanPrecision(0.0, 1.0)

    @test mean(dist1) === 0.0
    @test median(dist1) === 0.0
    @test mode(dist1) === 0.0
    @test weightedmean(dist1) === 0.0
    @test var(dist1) === 1.0
    @test std(dist1) === 1.0
    @test cov(dist1) === 1.0
    @test invcov(dist1) === 1.0
    @test precision(dist1) === 1.0
    @test entropy(dist1) ≈ 1.41893853320467
    @test pdf(dist1, 1.0) ≈ 0.24197072451914337
    @test pdf(dist1, -1.0) ≈ 0.24197072451914337
    @test pdf(dist1, 0.0) ≈ 0.3989422804014327
    @test logpdf(dist1, 1.0) ≈ -1.4189385332046727
    @test logpdf(dist1, -1.0) ≈ -1.4189385332046727
    @test logpdf(dist1, 0.0) ≈ -0.9189385332046728

    dist2 = NormalWeightedMeanPrecision(1.0, 1.0)

    @test mean(dist2) === 1.0
    @test median(dist2) === 1.0
    @test mode(dist2) === 1.0
    @test weightedmean(dist2) === 1.0
    @test var(dist2) === 1.0
    @test std(dist2) === 1.0
    @test cov(dist2) === 1.0
    @test invcov(dist2) === 1.0
    @test precision(dist2) === 1.0
    @test entropy(dist2) ≈ 1.41893853320467
    @test pdf(dist2, 1.0) ≈ 0.3989422804014327
    @test pdf(dist2, -1.0) ≈ 0.05399096651318806
    @test pdf(dist2, 0.0) ≈ 0.24197072451914337
    @test logpdf(dist2, 1.0) ≈ -0.9189385332046728
    @test logpdf(dist2, -1.0) ≈ -2.9189385332046727
    @test logpdf(dist2, 0.0) ≈ -1.4189385332046727

    dist3 = NormalWeightedMeanPrecision(1.0, 0.5)

    @test mean(dist3) === inv(0.5)
    @test median(dist3) === inv(0.5)
    @test mode(dist3) === inv(0.5)
    @test weightedmean(dist3) === 1.0
    @test var(dist3) === 2.0
    @test std(dist3) === sqrt(2.0)
    @test cov(dist3) === 2.0
    @test invcov(dist3) === inv(2.0)
    @test precision(dist3) === inv(2.0)
    @test entropy(dist3) ≈ 1.7655121234846454
    @test pdf(dist3, 1.0) ≈ 0.21969564473386122
    @test pdf(dist3, -1.0) ≈ 0.02973257230590734
    @test pdf(dist3, 0.0) ≈ 0.1037768743551487
    @test logpdf(dist3, 1.0) ≈ -1.5155121234846454
    @test logpdf(dist3, -1.0) ≈ -3.5155121234846454
    @test logpdf(dist3, 0.0) ≈ -2.2655121234846454
end

@testitem "NormalWeightedMeanPrecision: Base methods" begin
    include("./normal_family_setuptests.jl")

    @test convert(NormalWeightedMeanPrecision{Float32}, NormalWeightedMeanPrecision()) ==
          NormalWeightedMeanPrecision{Float32}(0.0f0, 1.0f0)
    @test convert(NormalWeightedMeanPrecision{Float64}, NormalWeightedMeanPrecision(0.0, 10.0)) ==
          NormalWeightedMeanPrecision{Float64}(0.0, 10.0)
    @test convert(NormalWeightedMeanPrecision{Float64}, NormalWeightedMeanPrecision(0.0, 0.1)) ==
          NormalWeightedMeanPrecision{Float64}(0.0, 0.1)
    @test convert(NormalWeightedMeanPrecision{Float64}, 0, 1) == NormalWeightedMeanPrecision{Float64}(0.0, 1.0)
    @test convert(NormalWeightedMeanPrecision{Float64}, 0, 10) == NormalWeightedMeanPrecision{Float64}(0.0, 10.0)
    @test convert(NormalWeightedMeanPrecision{Float64}, 0, 0.1) == NormalWeightedMeanPrecision{Float64}(0.0, 0.1)
    @test convert(NormalWeightedMeanPrecision, 0, 1) == NormalWeightedMeanPrecision{Float64}(0.0, 1.0)
    @test convert(NormalWeightedMeanPrecision, 0, 10) == NormalWeightedMeanPrecision{Float64}(0.0, 10.0)
    @test convert(NormalWeightedMeanPrecision, 0, 0.1) == NormalWeightedMeanPrecision{Float64}(0.0, 0.1)

    distribution = NormalWeightedMeanPrecision(-2.0, 3.0)

    @test distribution ≈ distribution
    @test distribution ≈ convert(NormalMeanPrecision, distribution)
    @test distribution ≈ convert(NormalMeanVariance, distribution)
end

@testitem "NormalWeightedMeanPrecision: vague" begin
    include("./normal_family_setuptests.jl")

    d1 = vague(NormalWeightedMeanPrecision)

    @test typeof(d1) <: NormalWeightedMeanPrecision
    @test mean(d1) == 0.0
    @test precision(d1) == 1e-12
end

@testitem "NormalWeightedMeanPrecision: prod" begin
    include("./normal_family_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
        @test prod(strategy, NormalWeightedMeanPrecision(-1, 1 / 1), NormalWeightedMeanPrecision(1, 1 / 1)) ≈
              NormalWeightedMeanPrecision(0, 2)
        @test prod(strategy, NormalWeightedMeanPrecision(-1, 1 / 2), NormalWeightedMeanPrecision(1, 1 / 4)) ≈
              NormalWeightedMeanPrecision(0, 3 / 4)
        @test prod(strategy, NormalWeightedMeanPrecision(2, 1 / 2), NormalWeightedMeanPrecision(0, 1 / 10)) ≈
              NormalWeightedMeanPrecision(2, 3 / 5)
    end
end
