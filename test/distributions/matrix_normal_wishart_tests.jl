
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

@testitem "MatrixNormalWishart: isproper (NaturalParametersSpace)" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    n, p = size(M)

    η_tup = MeanToNatural(MatrixNormalWishart)((M, U, V, ν))
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    @test isproper(NaturalParametersSpace(), MatrixNormalWishart, η, (n, p))

    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, η, nothing)

    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, fill(NaN, length(η)), (n, p))
    @test !isproper(NaturalParametersSpace(), MatrixNormalWishart, fill(Inf, length(η)), (n, p))
end

@testitem "MatrixNormalWishart: isproper (DefaultParametersSpace)" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0

    @test isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, U, V, ν))

    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, U, V, 0.5))
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, U, V, 1.0))

    V_bad = [1.0 5.0; 5.0 1.0]
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, U, V_bad, ν))

    U_bad = [1.0 5.0; 5.0 1.0]
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, U_bad, V, ν))

    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (fill(NaN, 2, 2), U, V, ν))
    @test !isproper(DefaultParametersSpace(), MatrixNormalWishart, (M, U, fill(Inf, 2, 2), ν))
end

@testitem "MatrixNormalWishart: MeanToNatural / NaturalToMean round-trip" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    n, p = size(M)

    η_tup = MeanToNatural(MatrixNormalWishart)((M, U, V, ν))
    η₁, η₂, η₃, η₄ = η_tup

    Ui = inv(U)
    @test η₁ ≈ Ui * M
    @test ExponentialFamily._mnw_smat(η₂, p) ≈ -1/2 * (inv(V) + M' * Ui * M)
    @test η₃ ≈ (ν + n - p - 1) / 2
    @test ExponentialFamily._mnw_smat(η₄, n) ≈ -1/2 * Ui

    Mr, Ur, Vr, νr = NaturalToMean(MatrixNormalWishart)(η_tup)
    @test Mr ≈ M
    @test Ur ≈ U
    @test Vr ≈ V
    @test νr ≈ ν
end

@testitem "MatrixNormalWishart: unpack_parameters" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    n, p = size(M)

    η_tup = MeanToNatural(MatrixNormalWishart)((M, U, V, ν))
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    η₁u, η₂u, η₃u, η₄u = unpack_parameters(MatrixNormalWishart, η, (n, p))

    @test η₁u ≈ η_tup[1]
    @test η₂u ≈ η_tup[2]
    @test η₃u ≈ η_tup[3]
    @test η₄u ≈ η_tup[4]

    Mr = [1.0 2.0 3.0; 4.0 5.0 6.0]
    Ur = [2.0 0.5; 0.5 1.5]
    Vr = diagm([1.0, 2.0, 3.0])
    νr = 5.0
    nr, pr = size(Mr)
    η_tup_r = MeanToNatural(MatrixNormalWishart)((Mr, Ur, Vr, νr))
    η_r = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup_r)
    η₁r, η₂r, η₃r, η₄r = unpack_parameters(MatrixNormalWishart, η_r, (nr, pr))
    @test η₁r ≈ η_tup_r[1]
    @test η₂r ≈ η_tup_r[2]
    @test η₃r ≈ η_tup_r[3]
    @test η₄r ≈ η_tup_r[4]
end

@testitem "MatrixNormalWishart: base measure (constant)" begin
    include("distributions_setuptests.jl")

    Λ = rand(Wishart(4, Matrix(1.0I, 2, 2)))
    X = rand(2, 2)
    x = (X, Λ)

    @test isbasemeasureconstant(MatrixNormalWishart) isa ConstantBaseMeasure

    bm_fn  = getbasemeasure(MatrixNormalWishart)
    lbm_fn = getlogbasemeasure(MatrixNormalWishart)

    @test bm_fn(x) ≈ 1.0
    @test lbm_fn(x) ≈ 0.0
end

@testitem "MatrixNormalWishart: getsufficientstatistics" begin
    include("distributions_setuptests.jl")

    X = [1.0 2.0; 3.0 4.0]
    Y = [2.0 0.5; 0.5 1.5]
    x = (X, Y)
    p = size(Y, 1)
    n = size(X, 1)

    T1, T2, T3, T4 = getsufficientstatistics(MatrixNormalWishart)

    @test T1(x) ≈ X * Y
    @test ExponentialFamily._mnw_smat(T2(x), p) ≈ Y
    @test T3(x) ≈ logdet(Y)
    @test ExponentialFamily._mnw_smat(T4(x), n) ≈ X * Y * X'
end

@testitem "MatrixNormalWishart: getlogpartition consistency" begin
    include("distributions_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    ν = 4.0
    n, p = size(M)

    η_tup = MeanToNatural(MatrixNormalWishart)((M, U, V, ν))
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    lp_nat = getlogpartition(NaturalParametersSpace(), MatrixNormalWishart, (n, p))(η)
    lp_def = getlogpartition(DefaultParametersSpace(), MatrixNormalWishart)((M, U, V, ν))

    @test lp_nat ≈ lp_def

    lp_expected = (n * p / 2) * log2π +
                  (p / 2) * logdet(U) +
                  (ν / 2) * logdet(V) +
                  (ν * p / 2) * log(2.0) +
                  logmvgamma(p, ν / 2)
    @test lp_def ≈ lp_expected
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
    @test all(x -> isposdef(x[2]), samples)

    mean_X = mean(x -> x[1], samples)
    @test mean_X ≈ M atol = 0.15

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
    @test getconditioner(ef) === nothing

    n, p = size(M)
    @test ExponentialFamily.getdims(ef) === (n, p)

    η₁, η₂, η₃, η₄ = ExponentialFamily.unpack_parameters(ef)

    Ui = inv(U)
    @test η₁ ≈ Ui * M
    @test ExponentialFamily._mnw_smat(η₂, p) ≈ -1/2 * (inv(V) + M' * Ui * M)
    @test η₃ ≈ (ν + n - p - 1) / 2
    @test ExponentialFamily._mnw_smat(η₄, n) ≈ -1/2 * Ui

    d_back = convert(Distribution, ef)
    @test d_back isa MatrixNormalWishart
    @test d_back.M ≈ M
    @test d_back.U ≈ U
    @test d_back.V ≈ V
    @test d_back.ν ≈ ν
end

@testitem "MatrixNormalWishart: logpdf via ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    M  = [1.0 2.0; 3.0 4.0]
    U  = [2.0 0.5; 0.5 1.5]
    V  = [1.5 0.3; 0.3 2.0]
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
    U_l = [2.0 0.5; 0.5 1.5]
    V_l = [1.5 0.3; 0.3 2.0]
    ν_l = 4.0
    left = MatrixNormalWishart(M_l, U_l, V_l, ν_l)

    M_r = [-1.0 0.5; 2.0 -1.0]
    U_r = [3.0 0.1; 0.1 2.0]
    V_r = [3.0 0.2; 0.2 1.0]
    ν_r = 5.0
    right = MatrixNormalWishart(M_r, U_r, V_r, ν_r)

    result = prod(PreserveTypeProd(Distribution), left, right)

    @test result isa MatrixNormalWishart
    n, p = size(M_l)

    Λl = inv(U_l)
    Λr = inv(U_r)
    Λ_expected = Λl + Λr
    U_expected = inv(Λ_expected)
    M_expected = U_expected * (Λl * M_l + Λr * M_r)
    Ω_expected = inv(V_l) + inv(V_r) + M_l' * Λl * M_l + M_r' * Λr * M_r - M_expected' * Λ_expected * M_expected
    V_expected = inv(Ω_expected)
    ν_expected = ν_l + ν_r + n - p - 1

    @test result.M ≈ M_expected
    @test result.U ≈ U_expected
    @test result.V ≈ V_expected
    @test result.ν ≈ ν_expected

    @test isproper(DefaultParametersSpace(), MatrixNormalWishart, (result.M, result.U, result.V, result.ν))

    xs = [rand(rng, result) for _ in 1:5]
    diffs = [logpdf(left, x) + logpdf(right, x) - logpdf(result, x) for x in xs]
    @test all(d -> isapprox(d, diffs[1]; atol = 1e-8), diffs)

    U = [2.0 0.5; 0.5 1.5]
    same_l = MatrixNormalWishart(M_l, U, V_l, ν_l)
    same_r = MatrixNormalWishart(M_r, U, V_r, ν_r)
    same_result = prod(PreserveTypeProd(Distribution), same_l, same_r)
    @test same_result.M ≈ (M_l + M_r) / 2
    @test same_result.U ≈ U / 2
    @test same_result.ν ≈ ν_l + ν_r + n - p - 1
end

@testitem "MatrixNormalWishart: test_exponentialfamily_interface" begin
    include("distributions_setuptests.jl")

    rng = StableRNG(42)
<<<<<<< HEAD
    Mr_22 = randn(rng, 2, 2)
    Ur_22 = randn(rng, 2, 2)
    Vr_22 = randn(rng, 2, 2)
    νr_22 = 100*rand(rng)+10
    Mr_32 = randn(rng, 3, 2)
    Ur_32 = randn(rng, 3, 3)
    Vr_32 = randn(rng, 2, 2)
    νr_32 = 100*rand(rng)+10
=======
    Mr_22 = randn(rng,2,2)
    Ur_22 = randn(rng,2,2)
    Vr_22 = randn(rng,2,2)
    νr_22 = 100*rand(rng)
    Mr_32 = randn(rng,3,2)
    Ur_32 = randn(rng,3,3)
    Vr_32 = randn(rng,2,2)
    νr_32 = 100*rand(rng)
>>>>>>> 58d748cde8a5cf2445a80161878247b956f8b105

    for (M, U, V, ν) in (
        (Mr_22, Ur_22 * Ur_22' + diagm(1e-8*ones(2)), Vr_22 * Vr_22' + diagm(1e-8*ones(2)), νr_22),
        (Mr_32, Ur_32 * Ur_32' + diagm(1e-8*ones(3)), Vr_32 * Vr_32' + diagm(1e-8*ones(2)), νr_32)
    )
        d = MatrixNormalWishart(M, U, V, ν)
        test_exponentialfamily_interface(d;
            test_parameters_conversion = false,
            test_basic_functions = false,
            test_fisherinformation_properties = false,
            test_fisherinformation_against_hessian = false,
            test_fisherinformation_against_jacobian = false,
            option_assume_no_allocations = false
        )
    end
end

@testitem "MatrixNormalWishart: Fisher information matches Hessian of logpartition" begin
    include("distributions_setuptests.jl")
    using ForwardDiff

    M = [0.2 -0.4; 1.1 0.7; -0.3 0.5]
    U = [1.8 0.2 0.1; 0.2 1.4 -0.3; 0.1 -0.3 1.1]
    V = [0.9 0.2; 0.2 1.3]
    ν = 6.5
    n, p = size(M)
    η_tup = MeanToNatural(MatrixNormalWishart)((M, U, V, ν))
    η = pack_parameters(NaturalParametersSpace(), MatrixNormalWishart, η_tup)

    F = getfisherinformation(NaturalParametersSpace(), MatrixNormalWishart, (n, p))(η)
    A = getlogpartition(NaturalParametersSpace(), MatrixNormalWishart, (n, p))
    H = ForwardDiff.hessian(A, η)

    @test size(F) == (length(η), length(η))
    @test F ≈ F'
    @test F ≈ H atol = 1e-6

    ef = convert(ExponentialFamilyDistribution, MatrixNormalWishart(M, U, V, ν))
    @test fisherinformation(ef) ≈ F
end
