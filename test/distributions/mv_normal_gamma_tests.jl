
@testitem "MvNormalGamma: common" begin
    include("distributions_setuptests.jl")

    μ = [0.5, -1.0]
    Λ = [2.0 0.3; 0.3 1.5]
    α, β = 3.0, 2.0
    dist = MvNormalGamma(μ, Λ, α, β)

    @test params(dist) == (μ, Λ, α, β)
    @test location(dist) == μ
    @test scale(dist) == Λ
    @test shape(dist) == α
    @test rate(dist) == β
    @test mean(dist) == (μ, α / β)
    @test length(dist) == 3
    @test eltype(dist) === Float64
end

@testitem "MvNormalGamma: type promotion" begin
    include("distributions_setuptests.jl")
    using ExponentialFamily: paramfloattype

    d1 = MvNormalGamma(rand(Float32, 2), [1.0 0.0; 0.0 1.0], 2.0, 3.0)
    @test paramfloattype(d1) == Float64

    d2 = MvNormalGamma(rand(Float32, 2), [1.0f0 0.0f0; 0.0f0 1.0f0], 2.0f0, 3.0f0)
    @test paramfloattype(d2) == Float32
end

@testitem "MvNormalGamma: logpdf factorizes as MvNormal × Gamma" begin
    include("distributions_setuptests.jl")

    rng = StableRNG(123)
    for d in (2, 3)
        A = randn(rng, d, d)
        Λ = A * A' + d * I
        μ = randn(rng, d)
        α, β = 2.0 + rand(rng), 1.0 + rand(rng)
        dist = MvNormalGamma(μ, Λ, α, β)

        for _ in 1:5
            τ = rand(rng, GammaShapeRate(α, β))
            x = rand(rng, MvNormalMeanPrecision(μ, τ * Λ))
            sample = vcat(x, τ)
            expected = logpdf(MvNormalMeanPrecision(μ, τ * Λ), x) + logpdf(GammaShapeRate(α, β), τ)
            @test logpdf(dist, sample) ≈ expected
            @test pdf(dist, sample) ≈ exp(expected)
        end
    end
end

@testitem "MvNormalGamma: ExponentialFamilyDistribution interface" begin
    include("distributions_setuptests.jl")

    rng = StableRNG(42)

    for d in (2, 3)
        A = randn(rng, d, d)
        Λ = A * A' + d * I
        μ = randn(rng, d)
        α = 2.0 + 10rand(rng)
        β = 1.0 + 10rand(rng)

        @testset let dist = MvNormalGamma(μ, Λ, α, β)
            ef = test_exponentialfamily_interface(
                dist;
                option_assume_no_allocations = false,
                # ForwardDiff comparisons are not meaningful here: `η₂` is stored as a free
                # `d×d` matrix while the family only depends on its symmetric part (the same
                # reason these are disabled for `MvNormalWishart`).
                test_gradlogpartition_properties = false,
                test_fisherinformation_against_hessian = false,
                test_fisherinformation_against_jacobian = false
            )

            # gradient equals E[sufficient statistics]; check via sampling without ForwardDiff
            run_test_gradlogpartition_properties(dist; test_against_forwardiff = false)

            (η1, η2, η3, η4) = unpack_parameters(MvNormalGamma, getnaturalparameters(ef))
            @test η1 ≈ Λ * μ
            @test η2 ≈ -Λ / 2
            @test η3 ≈ α + d / 2 - 1
            @test η4 ≈ -β - dot(μ, Λ, μ) / 2
        end
    end
end

@testitem "MvNormalGamma: isproper" begin
    include("distributions_setuptests.jl")

    # Valid natural parameters from a proper distribution
    d = MvNormalGamma([0.5, -1.0], [2.0 0.3; 0.3 1.5], 3.0, 2.0)
    η = getnaturalparameters(convert(ExponentialFamilyDistribution, d))
    @test isproper(NaturalParametersSpace(), MvNormalGamma, η, nothing)

    # Non-positive-definite Λ (η₂ = -Λ/2 must give posdef Λ)
    bad = copy(collect(η))
    bad[3] = 10.0  # corrupt an off/diagonal entry of vec(η₂) to break posdefness
    bad[4] = 10.0
    @test !isproper(NaturalParametersSpace(), MvNormalGamma, bad, nothing)

    # A conditioner is not supported
    @test !isproper(NaturalParametersSpace(), MvNormalGamma, η, [1.0])
    @test !isproper(DefaultParametersSpace(), MvNormalGamma, collect(η), [1.0])

    # NaN / Inf rejected
    @test !isproper(NaturalParametersSpace(), MvNormalGamma, [NaN, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0], nothing)
end

@testitem "MvNormalGamma: prod conjugacy" begin
    include("distributions_setuptests.jl")

    rng = StableRNG(7)
    for d in (2, 3)
        A1 = randn(rng, d, d)
        A2 = randn(rng, d, d)
        left = MvNormalGamma(randn(rng, d), A1 * A1' + d * I, 2.0 + rand(rng), 1.0 + rand(rng))
        right = MvNormalGamma(randn(rng, d), A2 * A2' + d * I, 2.0 + rand(rng), 1.0 + rand(rng))

        @test test_generic_simple_exponentialfamily_product(
            left,
            right,
            strategies = (
                ClosedProd(),
                GenericProd(),
                PreserveTypeProd(ExponentialFamilyDistribution),
                PreserveTypeProd(ExponentialFamilyDistribution{MvNormalGamma})
            )
        )
    end
end

@testitem "MvNormalGamma: reduces to NormalGamma when d == 1" begin
    include("distributions_setuptests.jl")

    μ, λ, α, β = 0.7, 1.4, 2.5, 1.1
    dmv = MvNormalGamma([μ], reshape([λ], 1, 1), α, β)
    dsc = NormalGamma(μ, λ, α, β)

    for xτ in ([0.3, 0.9], [-1.2, 2.0], [0.0, 0.5])
        @test logpdf(dmv, xτ) ≈ logpdf(dsc, xτ)
    end

    ηmv = getnaturalparameters(convert(ExponentialFamilyDistribution, dmv))
    ηsc = getnaturalparameters(convert(ExponentialFamilyDistribution, dsc))
    @test collect(ηmv) ≈ collect(ηsc)

    @test getfisherinformation(NaturalParametersSpace(), MvNormalGamma)(ηmv) ≈
        getfisherinformation(NaturalParametersSpace(), NormalGamma)(collect(ηsc))
end
