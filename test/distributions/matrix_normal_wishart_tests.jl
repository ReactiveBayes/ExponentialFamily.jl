
@testitem "MatrixNormalWishart: constructor and accessors" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    d = MatrixNormalWishart(M, U, V, ν)

    @test d isa MatrixNormalWishart
    @test params(d) == (M, U, V, ν)
    @test size(d) == (2, 2)
    @test eltype(d) === Float64
    @test dof(d) == ν
    @test location(d) == M
    @test mean(d) == (M, ν * V)
end

@testitem "MatrixNormalWishart: type promotion" begin
    include("distributions_setuptests.jl")

    Mf32 = Float32[1.0 2.0; 3.0 4.0]
    Uf64 = [2.0 0.5; 0.5 1.5]
    Vf64 = [1.5 0.3; 0.3 2.0]
    d = MatrixNormalWishart(Mf32, Uf64, Vf64, 4.0)
    @test eltype(d) === Float64

    Mf32 = Float32[1.0 2.0; 3.0 4.0]
    Uf32 = Float32[2.0 0.5; 0.5 1.5]
    Vf32 = Float32[1.5 0.3; 0.3 2.0]
    d32 = MatrixNormalWishart(Mf32, Uf32, Vf32, 4.0f0)
    @test eltype(d32) === Float32
end

@testitem "MatrixNormalWishart: separate_conditioner and join_conditioner" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0

    # params(d) = (M, U, V, ν); separate_conditioner extracts U as conditioner
    cparams, conditioner = ExponentialFamily.separate_conditioner(MatrixNormalWishart, (M, U, V, ν))
    @test cparams == (M, V, ν)
    @test conditioner === U

    joined = ExponentialFamily.join_conditioner(MatrixNormalWishart, cparams, conditioner)
    @test joined == (M, U, V, ν)
end

@testitem "MatrixNormalWishart: isproper (NaturalParametersSpace)" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    d = MatrixNormalWishart(M, U, V, ν)

    cparams, U_cond = ExponentialFamily.separate_conditioner(MatrixNormalWishart, params(d))
    η_tup = MeanToNatural(MatrixNormalWishart)(cparams, U_cond)
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    @test isproper(NaturalParametersSpace(), MatrixNormalWishart, η, U_cond)

    # Nothing conditioner → false (U is required)
    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, η, nothing)

    # NaN / Inf in packed η → false
    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, fill(NaN, length(η)), U_cond)
    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, fill(Inf, length(η)), U_cond)
end

@testitem "MatrixNormalWishart: isproper (DefaultParametersSpace)" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0

    @test isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, V, ν), U)

    # Nothing conditioner → false
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, V, ν), nothing)

    # ν ≤ p-1 = 1 → false  (p=2 here, so need ν > 1)
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, V, 0.5), U)
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, V, 1.0), U)

    # Non-PD V → false
    V_bad = [1.0 5.0; 5.0 1.0]  # not positive definite
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, V_bad, ν), U)

    # NaN / Inf in matrix params → false
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (fill(NaN, 2, 2), V, ν), U)
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, fill(Inf, 2, 2), ν), U)
end

@testitem "MatrixNormalWishart: MeanToNatural / NaturalToMean round-trip" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    n, p = size(M)

    cparams = (M, V, ν)
    η_tup = MeanToNatural(MatrixNormalWishart)(cparams, U)
    η₁, η₂, η₃ = η_tup

    # Manual verification of natural parameters
    Ui = inv(U)
    @test η₁ ≈ Ui * M
    @test η₂ ≈ -1/2 * (inv(V) + M' * Ui * M)
    @test η₃ ≈ (ν + n - p - 1) / 2

    # Round-trip
    Mr, Vr, νr = NaturalToMean(MatrixNormalWishart)(η_tup, U)
    @test Mr ≈ M
    @test Vr ≈ V
    @test νr ≈ ν
end

@testitem "MatrixNormalWishart: unpack_parameters" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0

    cparams = (M, V, ν)
    η_tup = MeanToNatural(MatrixNormalWishart)(cparams, U)
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    η₁u, η₂u, η₃u = unpack_parameters(MatrixNormalWishart, η, U)

    @test η₁u ≈ η_tup[1]
    @test η₂u ≈ η_tup[2]
    @test η₃u ≈ η_tup[3]

    # Rectangular: n=2, p=3
    Mr = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2×3
    Ur = [2.0 0.5; 0.5 1.5]           # 2×2 (conditioner)
    Vr = diagm([1.0, 2.0, 3.0])       # 3×3
    νr = 5.0
    cparams_r = (Mr, Vr, νr)
    η_tup_r = MeanToNatural(MatrixNormalWishart)(cparams_r, Ur)
    η_r = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup_r)
    η₁r, η₂r, η₃r = unpack_parameters(MatrixNormalWishart, η_r, Ur)
    @test η₁r ≈ η_tup_r[1]
    @test η₂r ≈ η_tup_r[2]
    @test η₃r ≈ η_tup_r[3]
end

@testitem "MatrixNormalWishart: base measure (non-constant)" begin
    include("distributions_setuptests.jl")

    U = [2.0 0.5; 0.5 1.5]
    Λ = rand(Wishart(4, Matrix(1.0I, 2, 2)))
    X = rand(2, 2)
    x = (X, Λ)

    @test isbasemeasureconstant(MatrixNormalWishart) isa NonConstantBaseMeasure

    # Log base measure should be -1/2 tr(X' U⁻¹ X Λ)
    bm_fn  = getbasemeasure(MatrixNormalWishart, U)
    lbm_fn = getlogbasemeasure(MatrixNormalWishart, U)

    lbm_expected = -1/2 * tr(X' * inv(U) * X * Λ)
    @test lbm_fn(x) ≈ lbm_expected
    @test bm_fn(x)  ≈ exp(lbm_expected)
end

@testitem "MatrixNormalWishart: getsufficientstatistics" begin
    include("distributions_setuptests.jl")

    U = [2.0 0.5; 0.5 1.5]
    X = [1.0 2.0; 3.0 4.0]
    Y = [2.0 0.5; 0.5 1.5]   # acts as precision sample
    x = (X, Y)

    T1, T2, T3 = getsufficientstatistics(MatrixNormalWishart, U)

    @test T1(x) ≈ X * Y
    @test T2(x) ≈ Y
    @test T3(x) ≈ logdet(Y)
end

@testitem "MatrixNormalWishart: getlogpartition consistency" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    n, p = size(M)

    # Natural-space log-partition
    cparams = (M, V, ν)
    η_tup = MeanToNatural(MatrixNormalWishart)(cparams, U)
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    lp_nat = getlogpartition(NaturalParametersSpace(), MatrixNormalWishart, U)(η)
    lp_def = getlogpartition(DefaultParametersSpace(), MatrixNormalWishart, U)((M, V, ν))

    # Both spaces must give the same normalization constant
    @test lp_nat ≈ lp_def

    # Explicit formula check for default-space LP
    lp_expected = (n * p / 2) * log2π +
                  (p / 2) * logdet(U) +
                  (ν / 2) * logdet(V) +
                  (ν * p / 2) * log(2.0) +
                  logmvgamma(p, ν / 2)
    @test lp_def ≈ lp_expected
end

@testitem "MatrixNormalWishart: getfisherinformation (DefaultParametersSpace)" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    n, p = size(M)

    fi_fn = getfisherinformation(DefaultParametersSpace(), MatrixNormalWishart, U)
    F = fi_fn((M, V, ν))

    total = n * p + p^2 + 1
    @test size(F) == (total, total)
    @test issymmetric(F)
    @test isposdef(F)

    # Diagonal blocks against analytic formulas
    Ui = inv(U)
    Vi = inv(V)
    np = n * p
    pp = p^2

    F_M = F[1:np, 1:np]
    F_V = F[(np+1):(np+pp), (np+1):(np+pp)]
    F_ν = F[np+pp+1, np+pp+1]

    @test F_M ≈ ν * kron(Vi, Ui)
    @test F_V ≈ (ν / 2) * kron(Vi, Vi)
    @test F_ν ≈ mvtrigamma(p, ν / 2) / 4
end

@testitem "MatrixNormalWishart: pdf and logpdf" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    d = MatrixNormalWishart(M, U, V, ν)

    rng = StableRNG(42)
    for _ in 1:5
        x = rand(rng, d)
        @test pdf(d, x) >= 0
        @test logpdf(d, x) ≈ log(pdf(d, x)) atol = 1e-10

        # Consistency with the factored form p(X|Y) p(Y)
        X, Y = x
        logpdf_factored = logpdf(MatrixNormal(M, U, cholinv(Y)), X) +
                          logpdf(Wishart(ν, V), Y)
        @test logpdf(d, x) ≈ logpdf_factored atol = 1e-10
    end
end

@testitem "MatrixNormalWishart: rand" begin
    include("distributions_setuptests.jl")

    M = zeros(2, 2)
    U = Matrix(1.0I, 2, 2)
    V = Matrix(1.0I, 2, 2)
    ν = 4.0
    d = MatrixNormalWishart(M, U, V, ν)

    rng = StableRNG(42)
    samples = rand(rng, d, 2000)

    @test length(samples) == 2000
    @test all(x -> x isa Tuple && length(x) == 2, samples)
    @test all(x -> isposdef(x[2]), samples)  # Y must be positive definite

    # Sample mean of X should be close to M = 0
    mean_X = mean(x -> x[1], samples)
    @test mean_X ≈ M atol = 0.15

    # Sample mean of Y should be close to ν * V = 4I
    mean_Y = mean(x -> x[2], samples)
    @test mean_Y ≈ ν * V atol = 0.5
end

@testitem "MatrixNormalWishart: convert to ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    d = MatrixNormalWishart(M, U, V, ν)

    ef = convert(ExponentialFamilyDistribution, d)

    @test ef isa ExponentialFamilyDistribution{MatrixNormalWishart}
    @test getconditioner(ef) === U

    # Packed natural parameters recover correct values via unpack
    η = getnaturalparameters(ef)
    η₁, η₂, η₃ = unpack_parameters(MatrixNormalWishart, η, U)

    Ui = inv(U)
    n, p = size(M)
    @test η₁ ≈ Ui * M
    @test η₂ ≈ -1/2 * (inv(V) + M' * Ui * M)
    @test η₃ ≈ (ν + n - p - 1) / 2

    # isproper on the ef object
    @test isproper(NaturalParametersSpace(), MatrixNormalWishart, η, U)
end

@testitem "MatrixNormalWishart: logpdf via ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    d  = MatrixNormalWishart(M, U, V, ν)
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

    M_l = [1.0 2.0; 3.0 4.0]
    U   = [2.0 0.5; 0.5 1.5]   # shared conditioner
    V_l = [1.5 0.3; 0.3 2.0]
    ν_l = 4.0
    left = MatrixNormalWishart(M_l, U, V_l, ν_l)

    M_r = [-1.0 0.5; 2.0 -1.0]
    V_r = [3.0 0.2; 0.2 1.0]
    ν_r = 5.0
    right = MatrixNormalWishart(M_r, U, V_r, ν_r)

    result = prod(PreserveTypeProd(Distribution), left, right)

    @test result isa MatrixNormalWishart
    n, p = size(M_l)

    # M is additive in natural param space: M_prod = M_l + M_r
    @test result.M ≈ M_l + M_r

    # ν accumulates with correction
    @test result.ν ≈ ν_l + ν_r + n - p - 1

    # Product should be a proper distribution
    @test isproper(DefaultParametersSpace(), MatrixNormalWishart, (result.M, result.V, result.ν), U)

    # Product in EF space should match direct parameter-space product
    ef_l = convert(ExponentialFamilyDistribution, left)
    ef_r = convert(ExponentialFamilyDistribution, right)
    ef_prod = prod(PreserveTypeProd(ExponentialFamilyDistribution{MatrixNormalWishart}), ef_l, ef_r)
    dist_from_ef = convert(Distribution, ef_prod)
    @test dist_from_ef isa MatrixNormalWishart
    @test dist_from_ef.M ≈ result.M atol = 1e-10
    @test dist_from_ef.V ≈ result.V atol = 1e-10
    @test dist_from_ef.ν ≈ result.ν atol = 1e-10
end
