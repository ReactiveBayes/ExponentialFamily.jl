
@testitem "MatrixNormalWishart: constructor and accessors" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]
    d  = MatrixNormalWishart(M₀, Ψ, κ, ν, V)

    @test d isa MatrixNormalWishart
    @test params(d) == (M₀, Ψ, κ, ν, V)
    @test size(d) == (2, 2)
    @test eltype(d) === Float64
    @test dof(d) == ν
    @test location(d) == M₀
    @test mean(d) == (M₀, ν * Ψ)
end

@testitem "MatrixNormalWishart: type promotion" begin
    include("distributions_setuptests.jl")

    M₀f32 = Float32[1.0 2.0; 3.0 4.0]
    Ψf64  = [2.0 0.5; 0.5 1.5]
    V     = [1.5 0.3; 0.3 2.0]
    d = MatrixNormalWishart(M₀f32, Ψf64, 2.0, 4.0, V)
    @test eltype(d) === Float64

    M₀f32 = Float32[1.0 2.0; 3.0 4.0]
    Ψf32  = Float32[2.0 0.5; 0.5 1.5]
    Vf32  = Float32[1.5 0.3; 0.3 2.0]
    d32 = MatrixNormalWishart(M₀f32, Ψf32, 2.0f0, 4.0f0, Vf32)
    @test eltype(d32) === Float32
end

@testitem "MatrixNormalWishart: separate_conditioner and join_conditioner" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]

    cparams, conditioner = ExponentialFamily.separate_conditioner(MatrixNormalWishart, (M₀, Ψ, κ, ν, V))
    @test cparams == (M₀, Ψ, κ, ν)
    @test conditioner === V

    joined = ExponentialFamily.join_conditioner(MatrixNormalWishart, cparams, conditioner)
    @test joined == (M₀, Ψ, κ, ν, V)
end

@testitem "MatrixNormalWishart: isproper (NaturalParametersSpace)" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]
    d  = MatrixNormalWishart(M₀, Ψ, κ, ν, V)

    cparams, V_cond = ExponentialFamily.separate_conditioner(MatrixNormalWishart, params(d))
    η_tup = MeanToNatural(MatrixNormalWishart)(cparams, V_cond)
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    @test isproper(NaturalParametersSpace(), MatrixNormalWishart, η, V_cond)

    # Nothing conditioner → false (V is required)
    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, η, nothing)

    # NaN / Inf in packed η → false
    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, fill(NaN, length(η)), V_cond)
    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, fill(Inf, length(η)), V_cond)
end

@testitem "MatrixNormalWishart: isproper (DefaultParametersSpace)" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]

    @test isproper(DefaultParametersSpace(), MatrixNormalWishart, (M₀, Ψ, κ, ν), V)

    # Nothing conditioner → false
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M₀, Ψ, κ, ν), nothing)

    # κ ≤ 0 → false
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M₀, Ψ, 0.0, ν), V)
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M₀, Ψ, -1.0, ν), V)

    # ν ≤ n-1 = 1 → false
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M₀, Ψ, κ, 0.5), V)

    # NaN / Inf in matrix params → false
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (fill(NaN, 2, 2), Ψ, κ, ν), V)
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M₀, fill(Inf, 2, 2), κ, ν), V)
end

@testitem "MatrixNormalWishart: MeanToNatural / NaturalToMean round-trip" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]

    cparams = (M₀, Ψ, κ, ν)
    η_tup = MeanToNatural(MatrixNormalWishart)(cparams, V)
    η₁, η₂, η₃, η₄ = η_tup

    # Manual verification of natural parameters
    Vi = inv(V)
    @test η₁ ≈ κ * M₀ * Vi
    @test η₂ ≈ -1/2 * (inv(Ψ) + κ * M₀ * Vi * M₀')
    @test η₃ ≈ -κ / 2
    @test η₄ ≈ (ν + size(V, 1) - size(M₀, 1) - 1) / 2

    # Round-trip
    M₀r, Ψr, κr, νr = NaturalToMean(MatrixNormalWishart)(η_tup, V)
    @test M₀r ≈ M₀
    @test Ψr  ≈ Ψ
    @test κr  ≈ κ
    @test νr  ≈ ν
end

@testitem "MatrixNormalWishart: unpack_parameters" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]

    cparams = (M₀, Ψ, κ, ν)
    η_tup = MeanToNatural(MatrixNormalWishart)(cparams, V)
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    η₁u, η₂u, η₃u, η₄u = unpack_parameters(MatrixNormalWishart, η, V)

    @test η₁u ≈ η_tup[1]
    @test η₂u ≈ η_tup[2]
    @test η₃u ≈ η_tup[3]
    @test η₄u ≈ η_tup[4]

    # Rectangular: n=2, p=3
    M₀r = [1.0 2.0 3.0; 4.0 5.0 6.0]
    Ψr  = [2.0 0.5; 0.5 1.5]
    Vr  = diagm([1.0, 2.0, 3.0])
    cparams_r = (M₀r, Ψr, κ, ν + 2)
    η_tup_r = MeanToNatural(MatrixNormalWishart)(cparams_r, Vr)
    η_r = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup_r)
    η₁r, η₂r, η₃r, η₄r = unpack_parameters(MatrixNormalWishart, η_r, Vr)
    @test η₁r ≈ η_tup_r[1]
    @test η₂r ≈ η_tup_r[2]
    @test η₃r ≈ η_tup_r[3]
    @test η₄r ≈ η_tup_r[4]
end

@testitem "MatrixNormalWishart: getbasemeasure" begin
    include("distributions_setuptests.jl")

    V = [1.5 0.3; 0.3 2.0]
    x = (rand(2, 2), rand(Wishart(4, Matrix(1.0I, 2, 2))))

    @test isbasemeasureconstant(MatrixNormalWishart) isa ConstantBaseMeasure
    @test getbasemeasure(MatrixNormalWishart)(x) == 1.0
    @test getbasemeasure(MatrixNormalWishart, V)(x) == 1.0
    @test getlogbasemeasure(MatrixNormalWishart)(x) == 0.0
    @test getlogbasemeasure(MatrixNormalWishart, V)(x) == 0.0
end

@testitem "MatrixNormalWishart: getsufficientstatistics" begin
    include("distributions_setuptests.jl")

    V  = [1.5 0.3; 0.3 2.0]
    M  = [1.0 2.0; 3.0 4.0]
    Λ  = [2.0 0.5; 0.5 1.5]   # acts as precision
    x  = (M, Λ)

    T1, T2, T3, T4 = getsufficientstatistics(MatrixNormalWishart, V)

    @test T1(x) ≈ Λ * M
    @test T2(x) ≈ Λ
    @test T3(x) ≈ tr(inv(V) * M' * Λ * M)
    @test T4(x) ≈ logdet(Λ)
end

@testitem "MatrixNormalWishart: getlogpartition consistency" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]
    n, p = size(M₀)

    # Natural-space log-partition
    cparams = (M₀, Ψ, κ, ν)
    η_tup = MeanToNatural(MatrixNormalWishart)(cparams, V)
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    lp_nat = getlogpartition(NaturalParametersSpace(), MatrixNormalWishart, V)(η)
    lp_def = getlogpartition(DefaultParametersSpace(), MatrixNormalWishart, V)((M₀, Ψ, κ, ν))

    # Both spaces must give the same normalization constant
    @test lp_nat ≈ lp_def

    # Explicit formula check for default-space LP
    lp_expected = (n * p / 2) * log2π +
                  (n / 2) * logdet(V) +
                  -(n * p / 2) * log(κ) +
                  (ν * n / 2) * log(2.0) +
                  (ν / 2) * logdet(Ψ) +
                  logmvgamma(n, ν / 2)
    @test lp_def ≈ lp_expected
end

@testitem "MatrixNormalWishart: getfisherinformation (DefaultParametersSpace)" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]
    n, p = size(M₀)

    fi_fn = getfisherinformation(DefaultParametersSpace(), MatrixNormalWishart, V)
    F = fi_fn((M₀, Ψ, κ, ν))

    total = n * p + n^2 + 2
    @test size(F) == (total, total)
    @test issymmetric(F)
    @test isposdef(F)

    # Diagonal blocks against analytic formulas
    Vi = inv(V)
    Ψi = inv(Ψ)
    np = n * p
    nn = n^2

    F_M₀ = F[1:np, 1:np]
    F_Ψ  = F[(np+1):(np+nn), (np+1):(np+nn)]
    F_κ  = F[np+nn+1, np+nn+1]
    F_ν  = F[np+nn+2, np+nn+2]

    @test F_M₀ ≈ ν * κ * kron(Vi, Ψi)
    @test F_Ψ  ≈ (ν / 2) * kron(Ψi, Ψi)
    @test F_κ  ≈ n * p / (2κ^2)
    @test F_ν  ≈ mvtrigamma(n, ν / 2) / 4
end

@testitem "MatrixNormalWishart: pdf and logpdf" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]
    d  = MatrixNormalWishart(M₀, Ψ, κ, ν, V)

    rng = StableRNG(42)
    for _ in 1:5
        x = rand(rng, d)
        @test pdf(d, x) >= 0
        @test logpdf(d, x) ≈ log(pdf(d, x)) atol = 1e-10

        # Consistency with the factored form p(M|Λ) p(Λ)
        M, Λ = x
        logpdf_factored = logpdf(MatrixNormal(M₀, cholinv(κ * Λ), V), M) +
                          logpdf(Wishart(ν, Ψ), Λ)
        @test logpdf(d, x) ≈ logpdf_factored atol = 1e-10
    end
end

@testitem "MatrixNormalWishart: rand" begin
    include("distributions_setuptests.jl")

    M₀ = zeros(2, 2)
    Ψ  = Matrix(1.0I, 2, 2)
    κ  = 1.0
    ν  = 4.0
    V  = Matrix(1.0I, 2, 2)
    d  = MatrixNormalWishart(M₀, Ψ, κ, ν, V)

    rng = StableRNG(42)
    samples = rand(rng, d, 2000)

    @test length(samples) == 2000
    @test all(x -> x isa Tuple && length(x) == 2, samples)
    @test all(x -> isposdef(x[2]), samples)  # Λ must be positive definite

    # Sample mean of M should be close to M₀ = 0
    mean_M = mean(x -> x[1], samples)
    @test mean_M ≈ M₀ atol = 0.15

    # Sample mean of Λ should be close to ν * Ψ = 4I
    mean_Λ = mean(x -> x[2], samples)
    @test mean_Λ ≈ ν * Ψ atol = 0.5
end

@testitem "MatrixNormalWishart: convert to ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]
    d  = MatrixNormalWishart(M₀, Ψ, κ, ν, V)

    ef = convert(ExponentialFamilyDistribution, d)

    @test ef isa ExponentialFamilyDistribution{MatrixNormalWishart}
    @test getconditioner(ef) === V

    # Packed natural parameters recover correct values via unpack
    η = getnaturalparameters(ef)
    η₁, η₂, η₃, η₄ = unpack_parameters(MatrixNormalWishart, η, V)

    Vi = inv(V)
    @test η₁ ≈ κ * M₀ * Vi
    @test η₂ ≈ -1/2 * (inv(Ψ) + κ * M₀ * Vi * M₀')
    @test η₃ ≈ -κ / 2
    @test η₄ ≈ (ν + size(V, 1) - size(M₀, 1) - 1) / 2

    # isproper on the ef object
    @test isproper(NaturalParametersSpace(), MatrixNormalWishart, η, V)
end

@testitem "MatrixNormalWishart: logpdf via ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    M₀ = [1.0 2.0; 3.0 4.0]
    Ψ  = [2.0 0.5; 0.5 1.5]
    κ  = 2.0
    ν  = 4.0
    V  = [1.5 0.3; 0.3 2.0]
    d  = MatrixNormalWishart(M₀, Ψ, κ, ν, V)
    ef = convert(ExponentialFamilyDistribution, d)

    rng = StableRNG(42)
    for _ in 1:5
        x = rand(rng, d)
        @test logpdf(ef, x) ≈ logpdf(d, x) atol = 1e-8
    end
end

@testitem "MatrixNormalWishart: prod" begin
    include("distributions_setuptests.jl")

    rng = StableRNG(42)

    M₀l = [1.0 2.0; 3.0 4.0]
    Ψl  = [2.0 0.5; 0.5 1.5]
    κl  = 1.5
    νl  = 4.0
    V   = [1.5 0.3; 0.3 2.0]
    left = MatrixNormalWishart(M₀l, Ψl, κl, νl, V)

    M₀r = [-1.0 0.5; 2.0 -1.0]
    Ψr  = [3.0 0.2; 0.2 1.0]
    κr  = 0.5
    νr  = 5.0
    right = MatrixNormalWishart(M₀r, Ψr, κr, νr, V)

    result = prod(PreserveTypeProd(Distribution), left, right)

    @test result isa MatrixNormalWishart
    n, p = size(M₀l)

    # κ is additive
    @test result.κ ≈ κl + κr

    # ν accumulates with correction
    @test result.ν ≈ νl + νr + p - n - 1

    # Posterior mean is precision-weighted average
    κ_post = κl + κr
    @test result.M₀ ≈ (κl * M₀l + κr * M₀r) / κ_post

    # The product distribution should be proper
    @test isproper(DefaultParametersSpace(), MatrixNormalWishart, (result.M₀, result.Ψ, result.κ, result.ν), V)
end
