
@testitem "MatrixNormal: Constructor" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    d = MatrixNormal(M, U, V)

    @test d isa MatrixNormal
    @test params(d) == (M, U, V)
    @test size(d) == (2, 2)
    @test eltype(d) === Float64

    # Rectangular
    M2 = [1.0 2.0 3.0; 4.0 5.0 6.0]
    U2 = [2.0 0.5; 0.5 1.5]
    V2 = diagm([1.0, 2.0, 3.0])
    @test size(MatrixNormal(M2, U2, V2)) == (2, 3)
end

@testitem "MatrixNormal: distrname" begin
    include("./normal_family_setuptests.jl")

    d = MatrixNormal([1.0 2.0; 3.0 4.0], [2.0 0.5; 0.5 1.5], [1.5 0.3; 0.3 2.0])
    @test ExponentialFamily.distrname(d) == "MatrixNormal"
end

@testitem "MatrixNormal: vague" begin
    include("./normal_family_setuptests.jl")

    d = vague(MatrixNormal, (2, 2))
    @test d isa MatrixNormal
    @test size(d) == (2, 2)
    @test mean(d) == zeros(2, 2)

    d2 = vague(MatrixNormal, (3, 4))
    @test size(d2) == (3, 4)
    @test mean(d2) == zeros(3, 4)
end

@testitem "MatrixNormal: Stats" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    d = MatrixNormal(M, U, V)
    n, p = size(d)

    @test mean(d) == M
    @test cov(d) ≈ kron(V, U)    # Distributions.cov returns the full np×np covariance kron(V,U)
    @test invcov(d)[1] ≈ inv(U)
    @test invcov(d)[2] ≈ inv(V)
end

@testitem "MatrixNormal: entropy" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    d = MatrixNormal(M, U, V)
    n, p = size(d)

    # H = n*p/2*log(2π) + p/2*logdet(U) + n/2*logdet(V) + n*p/2
    H_expected = n*p/2 * log(2π) + p/2 * logdet(U) + n/2 * logdet(V) + n*p/2
    @test entropy(d) ≈ H_expected

    # H equals the log-partition (normalization constant) plus n*p/2
    A_expected = n*p/2 * log(2π) + p/2 * logdet(U) + n/2 * logdet(V)
    @test entropy(d) ≈ A_expected + n*p/2

    # entropy is invariant to M
    d2 = MatrixNormal(zeros(n, p), U, V)
    @test entropy(d) ≈ entropy(d2)
end

@testitem "MatrixNormal: logpdf and pdf" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    d = MatrixNormal(M, U, V)
    n, p = size(d)

    # At X = mean, the quadratic term vanishes:
    # logpdf(d, M) = -n*p/2*log(2π) - p/2*logdet(U) - n/2*logdet(V)
    A = n*p/2 * log(2π) + p/2 * logdet(U) + n/2 * logdet(V)
    @test logpdf(d, M) ≈ -A

    # pdf ≥ 0; logpdf = log(pdf)
    for _ in 1:5
        X = rand(d)
        @test pdf(d, X) ≥ 0
        @test logpdf(d, X) ≈ log(pdf(d, X)) atol = 1e-10
    end

    # logpdf is maximised at the mean
    X_off = M .+ 0.5 * randn(n, p)
    @test logpdf(d, M) ≥ logpdf(d, X_off)

    # Rectangular case
    M2 = zeros(2, 3)
    U2 = diagm([1.0, 1.0])
    V2 = diagm([1.0, 1.0, 1.0])
    d2 = MatrixNormal(M2, U2, V2)
    n2, p2 = 2, 3
    A2 = n2*p2/2 * log(2π)        # logdet of identity matrices is 0
    @test logpdf(d2, M2) ≈ -A2
end

@testitem "MatrixNormal: log-partition (DefaultParametersSpace)" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    n, p = 2, 2

    # A(M,U,V) = n*p/2*log(2π) + p/2*logdet(U) + n/2*logdet(V)
    lp_fn = getlogpartition(DefaultParametersSpace(), MatrixNormal)
    A = lp_fn((M, U, V))
    A_expected = n*p/2 * log(2π) + p/2 * logdet(U) + n/2 * logdet(V)
    @test A ≈ A_expected

    # Log-partition does not depend on M
    @test lp_fn((zeros(n, p), U, V)) ≈ A_expected
end

@testitem "MatrixNormal: grad log-partition (DefaultParametersSpace)" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    n, p = 2, 2

    grad_fn = getgradlogpartition(DefaultParametersSpace(), MatrixNormal)
    gM, gU, gV = grad_fn((M, U, V))

    # ∂A/∂M = 0
    @test gM ≈ zeros(n, p)
    # ∂A/∂U = p/2 * U^{-1}
    @test gU ≈ (p/2) * inv(U)
    # ∂A/∂V = n/2 * V^{-1}
    @test gV ≈ (n/2) * inv(V)
end

@testitem "MatrixNormal: MeanToNatural" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    n, p = 2, 2

    η1, η2 = MeanToNatural(MatrixNormal)((M, U, V))

    # η1 = vec(U^{-1} M V^{-1})
    @test η1 ≈ vec(inv(U) * M * inv(V))

    # η2 = -1/2 * V^{-1} ⊗ U^{-1}  (column-major Kronecker convention)
    @test η2 ≈ -1/2 * kron(inv(V), inv(U))
end

@testitem "MatrixNormal: NaturalToMean round-trip" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    n, p = size(M)

    η = MeanToNatural(MatrixNormal)((M, U, V))
    M2, U2, V2 = NaturalToMean(MatrixNormal)(η, nothing, (n, p))

    # Mean is always recovered exactly
    @test M2 ≈ M

    # U and V are only identifiable up to a positive scalar c:
    # MN(M, U, V) = MN(M, cU, V/c) for any c > 0.
    # Verify the recovered distribution is equivalent by comparing logpdfs.
    d_orig      = MatrixNormal(M, U, V)
    d_recovered = MatrixNormal(M2, U2, V2)
    rng         = StableRNG(42)
    for _ in 1:5
        X = rand(rng, d_orig)
        @test logpdf(d_orig, X) ≈ logpdf(d_recovered, X) atol = 1e-10
    end
end

@testitem "MatrixNormal: isproper (NaturalParametersSpace)" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]

    η1, η2 = MeanToNatural(MatrixNormal)((M, U, V))

    @test isproper(NaturalParametersSpace(), MatrixNormal, (η1, η2), nothing)
    @test !isproper(NaturalParametersSpace(), MatrixNormal, (fill(NaN, length(η1)), η2), nothing)
    @test !isproper(NaturalParametersSpace(), MatrixNormal, (fill(Inf, length(η1)), η2), nothing)
    @test !isproper(NaturalParametersSpace(), MatrixNormal, (η1, η2), :some_conditioner)
end

@testitem "MatrixNormal: prod" begin
    include("./normal_family_setuptests.jl")

    Ml = [1.0 2.0; 3.0 4.0]
    Ul = [2.0 0.5; 0.5 1.5]
    Vl = [1.5 0.3; 0.3 2.0]
    left = MatrixNormal(Ml, Ul, Vl)

    Mr = [-1.0 0.5; 2.0 -1.0]
    Ur = [3.0 0.2; 0.2 1.0]
    Vr = [2.0 0.1; 0.1 1.5]
    right = MatrixNormal(Mr, Ur, Vr)

    result = prod(PreserveTypeProd(MvNormalMeanCovariance), left, right)

    @test result isa MvNormalMeanCovariance

    # Verify precision and mean analytically
    Ui_l, Vi_l = inv(Ul), inv(Vl)
    Ui_r, Vi_r = inv(Ur), inv(Vr)
    Λ_expected = kron(Vi_l, Ui_l) + kron(Vi_r, Ui_r)
    ξ_expected = vec(Ui_l * Ml * Vi_l) + vec(Ui_r * Mr * Vi_r)
    Σ_expected = inv(Λ_expected)
    μ_expected = Σ_expected * ξ_expected

    @test mean(result) ≈ μ_expected
    @test cov(result) ≈ Σ_expected

    # logpdf of product equals sum of logpdfs up to a constant:
    # check log-ratio is preserved across samples
    rng = StableRNG(42)
    X1 = rand(rng, left)
    X2 = rand(rng, left)
    lhs = logpdf(result, vec(X1)) - logpdf(result, vec(X2))
    rhs = (logpdf(left, X1) + logpdf(right, X1)) - (logpdf(left, X2) + logpdf(right, X2))
    @test lhs ≈ rhs atol = 1e-10

    # default_prod_rule dispatches to PreserveTypeProd(MvNormalMeanCovariance)
    @test BayesBase.prod(BayesBase.default_prod_rule(MatrixNormal, MatrixNormal), left, right) isa MvNormalMeanCovariance
end

@testitem "MatrixNormal: convert to MvNormalMeanCovariance" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    d = MatrixNormal(M, U, V)

    mv = convert(MvNormalMeanCovariance, d)

    @test mv isa MvNormalMeanCovariance
    @test mean(mv) ≈ vec(M)
    # Cov(vec(X)) = V ⊗ U  (column-major: kron(V, U))
    @test cov(mv) ≈ kron(V, U)

    # logpdf must agree: X ~ MN(M,U,V)  iff  vec(X) ~ N(vec(M), V⊗U)
    for _ in 1:5
        X = rand(d)
        @test logpdf(d, X) ≈ logpdf(mv, vec(X)) atol = 1e-10
    end
end

@testitem "MatrixNormal: sufficient statistics" begin
    include("./normal_family_setuptests.jl")

    X = [1.0 2.0; 3.0 4.0]

    T1, T2 = getsufficientstatistics(MatrixNormal)(X)

    @test T1 ≈ vec(X)
    @test T2 ≈ vec(X) * vec(X)'
    @test size(T2) == (4, 4)
end

@testitem "MatrixNormal: rand samples" begin
    include("./normal_family_setuptests.jl")

    M = zeros(2, 2)
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    d = MatrixNormal(M, U, V)
    n, p = size(d)

    rng = StableRNG(42)
    nsamples = 5000
    samples = [rand(rng, d) for _ in 1:nsamples]

    # Sample mean ≈ M = 0
    @test mean(samples) ≈ M atol = 0.1

    # For zero-mean MN(0, U, V): E[X X'] = tr(V) * U
    sample_row_cov = mean(s -> s * s', samples) / tr(V)
    @test sample_row_cov ≈ U atol = 0.15
end

@testitem "MatrixNormal: precision" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    d = MatrixNormal(M, U, V)

    # precision(d) is defined as invcov(d)
    prec = precision(d)
    ic = invcov(d)
    @test prec[1] ≈ ic[1]
    @test prec[2] ≈ ic[2]

    # Both match the matrix inverses
    @test prec[1] ≈ inv(U)
    @test prec[2] ≈ inv(V)
end

@testitem "MatrixNormal: isproper (DefaultParametersSpace)" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]

    # Non-nothing conditioner always returns false
    @test !isproper(DefaultParametersSpace(), MatrixNormal, (M, U, V), :some_conditioner)
    @test !isproper(DefaultParametersSpace(), MatrixNormal, (M, U, V), 1)

    # Valid parameters with nothing conditioner
    @test isproper(DefaultParametersSpace(), MatrixNormal, (M, U, V), nothing)

    # NaN or Inf in any parameter matrix returns false
    @test !isproper(DefaultParametersSpace(), MatrixNormal, (fill(NaN, 2, 2), U, V), nothing)
    @test !isproper(DefaultParametersSpace(), MatrixNormal, (M, fill(Inf, 2, 2), V), nothing)
    @test !isproper(DefaultParametersSpace(), MatrixNormal, (M, U, fill(NaN, 2, 2)), nothing)
end

@testitem "MatrixNormal: getbasemeasure" begin
    include("./normal_family_setuptests.jl")

    X = [1.0 2.0; 3.0 4.0]
    bm = getbasemeasure(MatrixNormal)

    # MatrixNormal has a constant base measure equal to oneunit of x
    @test isbasemeasureconstant(MatrixNormal) isa ConstantBaseMeasure
    @test bm(X) == oneunit(X)
end

@testitem "MatrixNormal: getnaturalparameters (DefaultParametersSpace)" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]

    η_fn = getnaturalparameters(DefaultParametersSpace(), MatrixNormal)
    η1, η2 = η_fn((M, U, V))

    # Results match MeanToNatural
    η1_ref, η2_ref = MeanToNatural(MatrixNormal)((M, U, V))
    @test η1 ≈ η1_ref
    @test η2 ≈ η2_ref

    # Explicit formula: η1 = vec(U⁻¹ M V⁻¹), η2 = -1/2 * kron(V⁻¹, U⁻¹)
    @test η1 ≈ vec(inv(U) * M * inv(V))
    @test η2 ≈ -1/2 * kron(inv(V), inv(U))
end

@testitem "MatrixNormal: getfisherinformation (DefaultParametersSpace)" begin
    include("./normal_family_setuptests.jl")

    M = [1.0 2.0; 3.0 4.0]
    U = [2.0 0.5; 0.5 1.5]
    V = [1.5 0.3; 0.3 2.0]
    n, p = 2, 2

    fi_fn = getfisherinformation(DefaultParametersSpace(), MatrixNormal)
    F_M, F_U, F_V = fi_fn((M, U, V))

    # F_M = kron(U⁻¹, V⁻¹)
    @test F_M ≈ kron(inv(U), inv(V))

    # F_U = n/2 * kron(U⁻¹, U⁻¹)
    @test F_U ≈ n/2 * kron(inv(U), inv(U))

    # F_V = p/2 * kron(V⁻¹, V⁻¹)
    @test F_V ≈ p/2 * kron(inv(V), inv(V))

    # Fisher information does not depend on M
    F_M2, F_U2, F_V2 = fi_fn((zeros(n, p), U, V))
    @test F_M ≈ F_M2
    @test F_U ≈ F_U2
    @test F_V ≈ F_V2

    # Each block is symmetric positive definite
    @test issymmetric(F_M)
    @test issymmetric(F_U)
    @test issymmetric(F_V)
    @test isposdef(F_M)
    @test isposdef(F_U)
    @test isposdef(F_V)
end
