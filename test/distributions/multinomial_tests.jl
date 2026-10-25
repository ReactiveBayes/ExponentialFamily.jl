@testitem "Multinomial: probvec" begin
    include("distributions_setuptests.jl")

    @test probvec(Multinomial(5, [1 / 3, 1 / 3, 1 / 3])) == [1 / 3, 1 / 3, 1 / 3]
    @test probvec(Multinomial(3, [0.2, 0.2, 0.4, 0.1, 0.1])) == [0.2, 0.2, 0.4, 0.1, 0.1]
    @test probvec(Multinomial(2, [0.5, 0.5])) == [0.5, 0.5]
end

@testitem "Multinomial: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(Multinomial)
    @test_throws MethodError vague(Multinomial, 4)

    vague_dist1 = vague(Multinomial, 5, 4)
    @test typeof(vague_dist1) <: Multinomial
    @test probvec(vague_dist1) == [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    vague_dist2 = vague(Multinomial, 3, 5)
    @test typeof(vague_dist2) <: Multinomial
    @test probvec(vague_dist2) == [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
end

@testitem "Multinomial: test_EF_interface" begin
    include("distributions_setuptests.jl")

    using StableRNGs

    rng = StableRNG(42)

    for n in 2:6
        for trials in 2:10
            @testset let d = Multinomial(trials, normalize!(rand(rng, n), 1))
                test_exponentialfamily_interface(d;
                    # Returning dense vectors/matrices allocates; buffered Fisher products are tested separately.
                    option_assume_no_allocations = false
                )
            end
        end
    end
end

@testitem "Product of Multinomial distributions" begin
    include("multinomial_setuptests.jl")

    using StableRNGs

    rng = StableRNG(42)

    @testset "prod" begin
        for n in 4:6
            pleft = rand(rng, n)
            pleft = pleft ./ sum(pleft)
            pright = rand(rng, n)
            pright = pright ./ sum(pright)
            left = Multinomial(n, pleft)
            right = Multinomial(n, pright)
            efleft = convert(ExponentialFamilyDistribution, left)
            efright = convert(ExponentialFamilyDistribution, right)
            prod_ef = prod(PreserveTypeProd(ExponentialFamilyDistribution), efleft, efright)
            d = vague(Multinomial, n, n)
            sample_space = multinomial_test_support(n, n)
            weights = [pdf(left, x) * pdf(right, x) for x in sample_space]
            weights ./= sum(weights)
            actual = pdf(prod_ef, sample_space)
            @test sum(actual) ≈ 1.0 rtol = 1e-12
            @test actual ≈ weights rtol = 1e-12
            @test actual ≈ pdf(prod(PreserveTypeProd(ExponentialFamilyDistribution), right, left), sample_space) rtol = 1e-12
            @test value_support(typeof(prod_ef)) === Discrete
            @test getconditioner(prod_ef) == n

            expected_mean = sum(w * x for (w, x) in zip(weights, sample_space))
            expected_cov = sum(w * (x - expected_mean) * (x - expected_mean)' for (w, x) in zip(weights, sample_space))
            @test gradlogpartition(prod_ef) ≈ expected_mean rtol = 1e-10
            @test fisherinformation(prod_ef) ≈ expected_cov rtol = 1e-10

            # Test basemeasure and sufficient statistics
            sample_x = rand(rng, d, 5)
            for xi in eachcol(sample_x)
                @test getbasemeasure(prod_ef)(xi) ≈ (factorial(n) / prod(@.factorial(xi)))^2 rtol = 1e-10
                @test sufficientstatistics(prod_ef, xi) == (xi,)
            end
        end

        # Test error cases for mismatched conditioners
        @test_throws AssertionError prod(
            PreserveTypeProd(ExponentialFamilyDistribution),
            convert(ExponentialFamilyDistribution, Multinomial(4, [0.2, 0.4, 0.4])),
            convert(ExponentialFamilyDistribution, Multinomial(5, [0.1, 0.3, 0.6]))
        )
        @test_throws AssertionError prod(
            PreserveTypeProd(ExponentialFamilyDistribution),
            convert(ExponentialFamilyDistribution, Multinomial(4, [0.2, 0.4, 0.4])),
            convert(ExponentialFamilyDistribution, Multinomial(3, [0.1, 0.3, 0.6]))
        )
    end
end

@testitem "Multinomial: natural parameters" begin
    include("distributions_setuptests.jl")

    @testset "natural parameters related " begin
        d1 = Multinomial(5, [0.1, 0.4, 0.5])
        d2 = Multinomial(5, [0.2, 0.4, 0.4])
        η1 = ExponentialFamilyDistribution(Multinomial, [log(0.1 / 0.5), log(0.4 / 0.5), 0.0], 5)
        η2 = ExponentialFamilyDistribution(Multinomial, [log(0.2 / 0.4), 0.0, 0.0], 5)

        @test convert(ExponentialFamilyDistribution, d1) ≈ η1
        @test convert(ExponentialFamilyDistribution, d2) ≈ η2

        @test convert(Distribution, η1) ≈ d1
        @test convert(Distribution, η2) ≈ d2

        @test logpartition(η1) ≈ -5log(0.5) rtol = 1e-14
        @test logpartition(η2) ≈ -5log(0.4) rtol = 1e-14

        @test basemeasure(η1, [1, 2, 2]) == 30.0
        @test basemeasure(η2, [1, 2, 2]) == 30.0

        @test logpdf(η1, [1, 2, 2]) ≈ logpdf(d1, [1, 2, 2]) atol = 1e-8
        @test logpdf(η2, [1, 2, 2]) ≈ logpdf(d2, [1, 2, 2]) atol = 1e-8

        @test pdf(η1, [1, 2, 2]) ≈ pdf(d1, [1, 2, 2]) atol = 1e-8
        @test pdf(η2, [1, 2, 2]) ≈ pdf(d2, [1, 2, 2]) atol = 1e-8
    end
end

@testitem "Multinomial: mean and covariance" begin
    include("distributions_setuptests.jl")

    using StableRNGs
    using Distributions: Dirichlet

    @testset "ExponentialFamilyDistribution mean,cov" begin
        rng = StableRNG(42)
        for n in 2:12
            p = rand(rng, Dirichlet(ones(n)))
            dist = Multinomial(n, p)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test cov(dist) ≈ cov(ef) atol = 1e-8
        end
    end
end

@testitem "Multinomial: stable derivatives and parameter types" begin
    include("distributions_setuptests.jl")

    for R in (Float32, Float64, BigFloat), η in (zeros(R, 3), R[1000, 999, -1000], R[-4, 1, 0])
        n = 7
        A = getlogpartition(NaturalParametersSpace(), Multinomial, n)
        g = getgradlogpartition(NaturalParametersSpace(), Multinomial, n)(η)
        F = getfisherinformation(NaturalParametersSpace(), Multinomial, n)(η)
        p = softmax(η)
        tolerance = 200eps(R)
        @test A(η) isa R
        @test eltype(g) === R
        @test eltype(F) === R
        @test all(isfinite, g)
        @test all(isfinite, F)
        @test g ≈ ForwardDiff.gradient(A, η) rtol = tolerance atol = tolerance
        @test F ≈ ForwardDiff.hessian(A, η) rtol = tolerance atol = tolerance
        @test F ≈ cov(Multinomial(n, p)) rtol = tolerance atol = tolerance
        @test F * ones(R, 3) ≈ zeros(R, 3) atol = tolerance
    end

    # Test the default-space extension away from the simplex as well as on it.
    for p in ([0.2, 0.3, 0.5], [2.0, 3.0, 5.0])
        n = 7
        A = getlogpartition(DefaultParametersSpace(), Multinomial, n)
        g = getgradlogpartition(DefaultParametersSpace(), Multinomial, n)(p)
        mapping = getmapping(DefaultParametersSpace() => NaturalParametersSpace(), Multinomial)
        η = mapping(p, n)
        J = ForwardDiff.jacobian(p -> mapping(p, n), p)
        F = getfisherinformation(DefaultParametersSpace(), Multinomial, n)(p)
        @test A(p) ≈ getlogpartition(NaturalParametersSpace(), Multinomial, n)(η)
        @test g ≈ ForwardDiff.gradient(A, p)
        @test g ≈ J' * getgradlogpartition(NaturalParametersSpace(), Multinomial, n)(η)
        @test F ≈ J' * getfisherinformation(NaturalParametersSpace(), Multinomial, n)(η) * J
        @test F * p ≈ zeros(3) atol = 1e-13
    end
end

@testitem "Multinomial: support, validity and degenerate cases" begin
    include("multinomial_setuptests.jl")

    ef = convert(ExponentialFamilyDistribution, Multinomial(5, [0.1, 0.4, 0.5]))
    for x in ([1, 2, 2], [1.0, 2.0, 2.0])
        @test insupport(ef, x)
        @test x in getsupport(ef)
    end
    for x in ([1, 4], [1, 1, 1], [-1, 2, 4], [0.5, 1.5, 3.0], [NaN, 2, 3], [Inf, 0, 0], 5)
        @test !insupport(ef, x)
    end
    for n in (-1, 1.0, nothing)
        @test !isproper(NaturalParametersSpace(), Multinomial, [0.0, 0.0], n)
        @test !isproper(DefaultParametersSpace(), Multinomial, [0.5, 0.5], n)
    end
    for η in (Float64[], [NaN, 0.0], [Inf, 0.0], [-Inf, -Inf])
        @test !isproper(NaturalParametersSpace(), Multinomial, η, 5)
    end
    for p in (Float64[], [NaN, 1.0], [-0.1, 1.1], [0.0, 0.0], [1.0, 1.0])
        @test !isproper(DefaultParametersSpace(), Multinomial, p, 5)
    end
    @test isproper(NaturalParametersSpace(), Multinomial, [-Inf, 0.0], 0)
    @test isproper(DefaultParametersSpace(), Multinomial, [0.0, 1.0], 0)

    for n in (0, 1, 5), p in ([1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.4, 0.6], [0.2, 0.8, 0.0])
        d = Multinomial(n, p)
        ef = convert(ExponentialFamilyDistribution, d)
        @test convert(Distribution, ef) ≈ d
        @test isproper(ef)
        for x in multinomial_test_support(n, length(p))
            @test insupport(ef, x)
            @test logpdf(ef, x) ≈ logpdf(d, x) atol = 1e-13
            @test pdf(ef, x) ≈ pdf(d, x) atol = 1e-13
        end
        @test gradlogpartition(ef) ≈ mean(d)
        @test fisherinformation(ef) ≈ cov(d) atol = 1e-13
        @test fishermul(ef, ones(length(p))) ≈ zeros(length(p)) atol = 1e-13
    end
    @test_throws ArgumentError getfisherinformation(DefaultParametersSpace(), Multinomial, 5)([0.0, 1.0])

    for n in (21, 200, 1000)
        d = Multinomial(n, [0.25, 0.75])
        ef = convert(ExponentialFamilyDistribution, d)
        x = [1, n - 1]
        @test basemeasure(ef, x) ≈ n rtol = 1e-11
        @test logpdf(ef, x) ≈ logpdf(d, x) rtol = 1e-11
    end
    @test getbasemeasure(Multinomial, 100)(Float32[1, 99]) ≈ 100
end

@testitem "Multinomial products: exact partition, boundaries and derivatives" begin
    include("multinomial_setuptests.jl")
    using ExponentialFamily: __compute_logpartition_multinomial_product
    using LogExpFunctions: logsumexp

    for n in (0, 1, 4), K in (1, 2, 3, 4)
        support = multinomial_test_support(n, K)
        A = __compute_logpartition_multinomial_product(K, n)
        for η in (zeros(K), collect(range(-2.0; step = 0.75, length = K)), collect(range(999.0; step = 1.0, length = K)))
            # Independent finite-sum reference, not the production recurrence.
            terms = [2log(Float64(factorial(big(n)) ÷ prod(factorial.(big.(x))))) + dot(η, x) for x in support]
            expected_A = logsumexp(terms)
            weights = exp.(terms .- expected_A)
            μ = sum(w * x for (w, x) in zip(weights, support))
            covariance = sum(w * (x - μ) * (x - μ)' for (w, x) in zip(weights, support))
            @test A(η) ≈ expected_A atol = 1e-11
            @test ForwardDiff.gradient(A, η) ≈ μ atol = 1e-10
            @test ForwardDiff.hessian(A, η) ≈ covariance atol = 1e-10
            @test A(η .+ 10) ≈ A(η) + 10n atol = 1e-10
        end
    end

    # Vandermonde's identity gives an independent large-trial check for K = 2.
    for n in (21, 200)
        A = __compute_logpartition_multinomial_product(2, n)
        @test A(zeros(2)) ≈ log(Float64(binomial(big(2n), big(n)))) rtol = 1e-12
        @test A([1000.0, 1000.0]) ≈ A(zeros(2)) + 1000n rtol = 1e-12
    end
    for R in (Float32, Float64, BigFloat)
        A = __compute_logpartition_multinomial_product(3, 4)
        η = R[0, 0, 0]
        @test A(η) isa R
        @test eltype(ForwardDiff.gradient(A, η)) === R
        @test eltype(ForwardDiff.hessian(A, η)) === R
        @test A(η) ≈ log(R(639)) rtol = 200eps(R)
        before = A(η)
        A(R[10, -2, 3])
        @test A(η) == before
    end
    @test_throws DimensionMismatch __compute_logpartition_multinomial_product(3, 4)(zeros(2))
    @test_throws ArgumentError __compute_logpartition_multinomial_product(0, 4)
    @test_throws ArgumentError __compute_logpartition_multinomial_product(2, -1)
    @test_throws ArgumentError __compute_logpartition_multinomial_product(2, 4)([NaN, 0.0])

    strategy = PreserveTypeProd(ExponentialFamilyDistribution)
    # The raw squared base measure overflows here, but logpdf must remain finite.
    large_product = prod(strategy, Multinomial(600, [0.5, 0.5]), Multinomial(600, [0.5, 0.5]))
    expected_logpdf = Float64(2log(binomial(big(600), big(300))) - log(binomial(big(1200), big(600))))
    @test isfinite(logbasemeasure(large_product, [300, 300]))
    @test logpdf(large_product, [300, 300]) ≈ expected_logpdf atol = 1e-10
    @test_throws DimensionMismatch prod(strategy, Multinomial(3, [0.5, 0.5]), Multinomial(3, [0.2, 0.3, 0.5]))
    @test_throws ArgumentError prod(strategy, Multinomial(3, [1.0, 0.0]), Multinomial(3, [0.0, 1.0]))
    for n in (0, 3)
        q = prod(strategy, Multinomial(n, [0.0, 1.0]), Multinomial(n, [0.4, 0.6]))
        @test pdf(q, [0, n]) ≈ 1
        @test isfinite(logpdf(q, [0, n]))
        @test !insupport(q, [n + 1, 0])
        @test !insupport(q, [n])
        @test !insupport(q, [-1, n + 1])
        n > 0 && @test pdf(q, [n, 0]) == 0
    end
    zero_trials = prod(strategy, Multinomial(0, [1.0, 0.0]), Multinomial(0, [0.0, 1.0]))
    @test pdf(zero_trials, [0, 0]) == 1
end
