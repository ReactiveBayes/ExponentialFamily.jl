@testitem "Fisher products: built-in kernels and allocation contract" begin
    # `test_exponentialfamily_interface` already runs `run_test_fishermul` for generic
    # numerical/API correctness. This separate test covers guarantees that depend on
    # selected types, sizes, and repeated calls: zero allocations for Float32/Float64,
    # small and BLAS factorization paths, buffer reuse after changing η and v, and
    # StaticArrays behavior. Keeping these cases here avoids adding all these variants
    # to every distribution fixture in the shared interface tests.

    include("distributions/distributions_setuptests.jl")
    using ExponentialFamily: WishartFast, InverseWishartFast
    using BenchmarkTools: @ballocated
    using StaticArrays

    rng = StableRNG(812)
    scalar_distributions = (
        Bernoulli(0.3), Beta(2.0, 3.0), Binomial(7, 0.3), Chisq(4.0),
        Erlang(3, 2.0), Exponential(2.0), GammaShapeRate(2.0, 3.0),
        GammaInverse(3.0, 2.0), Geometric(0.3), Laplace(1.0, 2.0),
        LogNormal(0.5, 1.2), NegativeBinomial(3, 0.4), NormalMeanVariance(0.7, 1.3),
        NormalGamma(0.4, 2.0, 3.0, 2.0), Pareto(3.0, 2.0), Poisson(2.0),
        Rayleigh(2.0), VonMises(0.4, 2.0), Weibull(2.0, 3.0)
    )
    for d in scalar_distributions
        run_test_fishermul(d)
    end

    for n in (1, 3, 24)
        A = randn(rng, n, n)
        S = A * A' + n * I
        μ = randn(rng, n)
        M = randn(rng, n, 2)
        distributions = (
            MvNormalMeanCovariance(μ, S), MvNormalMeanScalePrecision(μ, 2.0),
            WishartFast(n + 4.0, S), InverseWishartFast(n + 4.0, S),
            MvNormalGamma(μ, S, 3.0, 2.0), MvNormalWishart(μ, S, 2.0, n + 4.0),
            MatrixNormal(M, S, [2.0 0.2; 0.2 1.5]),
            MatrixNormalWishart(M, S, [2.0 0.2; 0.2 1.5], 5.0)
        )
        for d in distributions
            # These families' existing EF interface requires matrix dimension >= 2.
            n == 1 && d isa Union{WishartFast, InverseWishartFast, MvNormalWishart} && continue
            n <= 3 && run_test_fishermul(d)
        end
        # Large reference matrices are unnecessary for testing workspace allocation.
        for d in distributions, R in (Float32, Float64)
            n == 1 && d isa Union{WishartFast, InverseWishartFast, MvNormalWishart} && continue
            ef0 = convert(ExponentialFamilyDistribution, d)
            T = ExponentialFamily.exponential_family_typetag(d)
            η = R.(getnaturalparameters(ef0))
            ef = ExponentialFamilyDistribution(T, η, getconditioner(ef0), getattributes(ef0))
            v = randn(rng, R, length(η))
            w = similar(v)
            buffer = @inferred FisherMulBuffer(ef, v)
            fishermul!(w, ef, v, buffer)
            @test_no_allocations fishermul!(w, ef, v, buffer)
            if n <= 3
                @test w ≈ fisherinformation(ef) * v rtol = 50eps(R) atol = 50eps(R)
                # Recompute after changing both η and v; retain the same workspace.
                η2 = R(1.01) .* η
                v2 = randn(rng, R, length(v))
                fill!(w, R(NaN))
                fishermul!(w, ef, η2, v2, buffer)
                @test w ≈ fisherinformation(ef, η2) * v2 rtol = 100eps(R) atol = 100eps(R)
                @test_no_allocations fishermul!(w, ef, η2, v2, buffer)
            end
        end
    end

    simple = (Categorical([0.2, 0.3, 0.5]), Dirichlet([2.0, 3.0, 4.0]),
        DirichletCollection(reshape(collect(1.0:12.0), 3, 2, 2)),
        VonMisesFisher(normalize([1.0, 2.0, 3.0]), 2.0))
    for d in (scalar_distributions..., simple...), R in (Float32, Float64)
        ef0 = convert(ExponentialFamilyDistribution, d)
        T = ExponentialFamily.exponential_family_typetag(d)
        ef = ExponentialFamilyDistribution(T, R.(getnaturalparameters(ef0)), getconditioner(ef0), getattributes(ef0))
        v = randn(rng, R, length(getnaturalparameters(ef)))
        w = similar(v)
        buffer = FisherMulBuffer(ef, v)
        fishermul!(w, ef, v, buffer)
        @test w ≈ fisherinformation(ef) * v rtol = 100eps(R) atol = 100eps(R)
        if VERSION < v"1.12" && d isa Union{VonMises, VonMisesFisher}
            # SpecialFunctions.besseli allocates a 16-byte Ref per call on Julia
            # 1.10/1.11: four calls for VonMises, two for VonMisesFisher.
            allocation_budget = d isa VonMises ? 64 : 32
            @test (@ballocated fishermul!($w, $ef, $v, $buffer) samples = 1 evals = 1) <= allocation_budget
        else
            @test_no_allocations fishermul!(w, ef, v, buffer)
        end
    end
    for d in simple
        run_test_fishermul(d)
    end

    ef = convert(ExponentialFamilyDistribution, Beta(2.0, 3.0))
    v = SA[1.0, -2.0]
    @test fishermul(ef, v) isa SVector
    @test_no_allocations fishermul(ef, v)
end

@testitem "Fisher products: directions, views, and validation" begin
    include("distributions/distributions_setuptests.jl")
    using StaticArrays

    ef = convert(ExponentialFamilyDistribution, Categorical([0.2, 0.3, 0.5]))
    η = getnaturalparameters(ef)
    @test fishermul(ef, ones(3)) ≈ zeros(3) atol = 1e-15
    @test fishermul(ef, SA[1.0, 2.0, -1.0]) ≈ fisherinformation(ef) * [1.0, 2.0, -1.0]
    @test fishermul(ef, [1000.0, 1001.0, 1002.0], ones(3)) ≈ zeros(3) atol = 1e-15

    # Disjoint views and mixed precision preserve their inputs.
    data = [1.0, 2.0, -3.0, NaN, NaN, NaN]
    v = view(data, 1:3)
    w = view(data, 4:6)
    buffer = FisherMulBuffer(ef, v)
    @test fishermul!(w, ef, v, buffer) === w
    @test w ≈ fisherinformation(ef) * v
    @test v == [1.0, 2.0, -3.0]
    v32 = Float32[1, 2, -3]
    buffer32 = FisherMulBuffer(ef, v32)
    @test fishermul!(w, ef, v32, buffer32) ≈ fisherinformation(ef) * v32
    @test_no_allocations fishermul!(w, ef, v32, buffer32)

    @test_throws DimensionMismatch fishermul(ef, ones(2))
    @test_throws DimensionMismatch getfishermul(ef)(η, ones(4))
    @test_throws DimensionMismatch getfishermul(NaturalParametersSpace(), Categorical, 3.0)(η, ones(3))
    @test_throws DimensionMismatch fishermul!(zeros(2), ef, ones(3))
    @test_throws ArgumentError fishermul!(v, ef, v, buffer)
    @test_throws ArgumentError fishermul!(η, ef, v, buffer)
    @test_throws ArgumentError fishermul!(view(data, 2:4), ef, v, buffer)
    @test_throws MethodError fishermul(DefaultParametersSpace(), ef, v)
    @test_throws MethodError getfishermul!(DefaultParametersSpace(), ef)
    @test_throws MethodError FisherMulBuffer(DefaultParametersSpace(), ef, v)
    other = convert(ExponentialFamilyDistribution, Dirichlet([2.0, 3.0, 4.0]))
    @test_throws ArgumentError fishermul!(w, ef, v, FisherMulBuffer(other, v))
    larger = convert(ExponentialFamilyDistribution, Categorical(fill(0.25, 4)))
    @test_throws DimensionMismatch fishermul!(w, ef, v, FisherMulBuffer(larger, ones(4)))
    ef32 = ExponentialFamilyDistribution(Categorical, Float32.(η), 3, nothing)
    @test_throws ArgumentError fishermul!(w, ef, v, FisherMulBuffer(ef32, v32))

    # Skew-symmetric packed matrix directions must not be silently symmetrized.
    for d in (MvNormalMeanCovariance([0.4, -0.2], [2.0 0.3; 0.3 1.5]),
        MvNormalGamma([0.4, -0.2], [2.0 0.3; 0.3 1.5], 3.0, 2.0),
        MvNormalWishart([0.4, -0.2], [2.0 0.3; 0.3 1.5], 2.0, 6.0))
        q = convert(ExponentialFamilyDistribution, d)
        direction = zeros(length(getnaturalparameters(q)))
        direction[4], direction[5] = 1, -1
        @test fishermul(q, direction) ≈ fisherinformation(q) * direction
    end
end

@testitem "Fisher products: extension fallback" begin
    using ExponentialFamily, Distributions, LinearAlgebra
    struct FisherFallbackDistribution <: ContinuousUnivariateDistribution end
    ExponentialFamily.getfisherinformation(::NaturalParametersSpace, ::Type{FisherFallbackDistribution}) = η -> [η[1]+2 0.2; 0.2 η[2]+3]
    ef = ExponentialFamilyDistribution(FisherFallbackDistribution, [1.0, 2.0], nothing, nothing)
    v = [2.0, -1.0]
    w = similar(v)
    @test fishermul(ef, v) ≈ fisherinformation(ef) * v
    @test fishermul!(w, ef, v) === w
    @test w ≈ fisherinformation(ef) * v
    @test_throws ArgumentError FisherMulBuffer(ef, v)

    # The optional-nothing forwarding must honor two-argument getter extensions.
    struct FisherGetterDistribution <: ContinuousUnivariateDistribution end
    ExponentialFamily.getfishermul(::NaturalParametersSpace, ::Type{FisherGetterDistribution}) = (η, v) -> η .* v
    ExponentialFamily.getfishermul!(::NaturalParametersSpace, ::Type{FisherGetterDistribution}) = (w, η, v, buffer = nothing) -> (w .= η .* v)
    q = ExponentialFamilyDistribution(FisherGetterDistribution, [2.0, 3.0], nothing, nothing)
    @test fishermul(q, v) == [4.0, -3.0]
    @test fishermul!(w, q, v) == [4.0, -3.0]

    struct FisherAttributesDistribution <: ContinuousUnivariateDistribution end
    ExponentialFamily.getfisherinformation(::ExponentialFamilyDistribution{FisherAttributesDistribution}) = η -> [η[1]+4 0.1; 0.1 η[2]+5]
    attributes = ExponentialFamilyDistributionAttributes(x -> 1, (identity, abs2), η -> 0, nothing)
    q = ExponentialFamilyDistribution(FisherAttributesDistribution, [1.0, 2.0], nothing, attributes)
    @test fishermul(q, v) ≈ fisherinformation(q) * v
    @test fishermul!(w, q, v) ≈ fisherinformation(q) * v
    @test_throws ArgumentError FisherMulBuffer(q, v)
end

@testitem "Fisher products: storage growth and scalar reductions" begin
    using ExponentialFamily, Distributions, LinearAlgebra

    function product_allocations(d)
        q = convert(ExponentialFamilyDistribution, d)
        v = ones(length(getnaturalparameters(q)))
        fishermul(q, v)
        bytes = @allocated fishermul(q, v)
        return bytes, Base.summarysize(FisherMulBuffer(q, v))
    end
    for make_distribution in (
        n -> Dirichlet(fill(2.0, n)),
        n -> MvNormalMeanCovariance(fill(0.3, n), Matrix{Float64}(I, n, n))
    )
        small = product_allocations(make_distribution(8))
        large = product_allocations(make_distribution(32))
        # Packed Gaussian length grows approximately 16x, whereas a Fisher
        # matrix would grow approximately 256x. Allow fixed allocation overhead.
        bound = make_distribution(8) isa Dirichlet ? 6 : 20
        @test large[1] < bound * small[1]
        @test large[2] < bound * small[2]
    end

    normal = convert(ExponentialFamilyDistribution, NormalMeanVariance(0.4, 1.7))
    mvnormal = convert(ExponentialFamilyDistribution, MvNormalMeanCovariance([0.4], fill(1.7, 1, 1)))
    @test fishermul(normal, [0.2, -0.6]) ≈ fishermul(mvnormal, [0.2, -0.6])
    gamma = convert(ExponentialFamilyDistribution, NormalGamma(0.4, 1.7, 3.0, 2.0))
    mvgamma = convert(ExponentialFamilyDistribution, MvNormalGamma([0.4], fill(1.7, 1, 1), 3.0, 2.0))
    @test fishermul(gamma, [0.2, -0.6, 0.8, 0.1]) ≈ fishermul(mvgamma, [0.2, -0.6, 0.8, 0.1])
end
