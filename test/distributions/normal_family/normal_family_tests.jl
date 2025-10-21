
@testitem "NormalFamily: Univariate conversions" begin
    include("./normal_family_setuptests.jl")

    types  = union_types(UnivariateNormalDistributionsFamily{Float64})
    etypes = union_types(UnivariateNormalDistributionsFamily)

    rng = MersenneTwister(1234)

    for type in types
        left = convert(type, rand(rng, Float64), rand(rng, Float64))

        for type in [types..., etypes...]
            right = convert(type, left)
            check_basic_statistics(left, right)

            p1 = prod(PreserveTypeLeftProd(), left, right)
            @test typeof(p1) <: typeof(left)

            p2 = prod(PreserveTypeRightProd(), left, right)
            @test typeof(p2) <: typeof(right)

            for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
                p3 = prod(strategy, left, right)

                check_basic_statistics(p1, p2)
                check_basic_statistics(p2, p3)
                check_basic_statistics(p1, p3)
            end
        end
    end
end

@testitem "NormalFamily: Multivariate conversions" begin
    include("./normal_family_setuptests.jl")

    types  = union_types(MultivariateNormalDistributionsFamily{Float64})
    etypes = union_types(MultivariateNormalDistributionsFamily)

    dims = (2, 3, 5)
    rng  = MersenneTwister(1234)

    for dim in dims
        for left_type in types
            left = if (left_type <: MvNormalMeanScalePrecision)
                # `MvNormalMeanScalePrecision` cannot be constructed using a Matrix
                # it requires a scale parameter instead
                convert(left_type, rand(rng, Float64, dim), rand(rng, Float64))
            else
                convert(left_type, rand(rng, Float64, dim), Matrix(Diagonal(abs.(rand(rng, Float64, dim)))))
            end

            for right_type in [types..., etypes...]

                # It's not always possible to convert other multivariate types to `MvNormalMeanScalePrecision`
                # so we skip it here
                if (right_type <: MvNormalMeanScalePrecision)
                    continue
                end

                right = convert(right_type, left)
                check_basic_statistics(left, right)

                p1 = if !(left_type <: MvNormalMeanScalePrecision)
                    prod(PreserveTypeLeftProd(), left, right)
                else
                    nothing
                end

                p2 = if !(right_type <: MvNormalMeanScalePrecision)
                    prod(PreserveTypeRightProd(), left, right)
                else
                    nothing
                end

                if !isnothing(p1)
                    @test typeof(p1) <: typeof(left)
                end

                if !isnothing(p2)
                    @test typeof(p2) <: typeof(right)
                end

                for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
                    p3 = prod(strategy, left, right)

                    if !isnothing(p1) && !isnothing(p2)
                        check_basic_statistics(p1, p2)
                    end

                    if !isnothing(p2)
                        check_basic_statistics(p2, p3)
                    end

                    if !isnothing(p1)
                        check_basic_statistics(p1, p3)
                    end
                end
            end
        end
    end
end

@testitem "NormalFamily: Variate forms promotions" begin
    include("./normal_family_setuptests.jl")

    @test promote_variate_type(Univariate, NormalMeanVariance) === NormalMeanVariance
    @test promote_variate_type(Univariate, NormalMeanPrecision) === NormalMeanPrecision
    @test promote_variate_type(Univariate, NormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

    @test promote_variate_type(Multivariate, NormalMeanVariance) === MvNormalMeanCovariance
    @test promote_variate_type(Multivariate, NormalMeanPrecision) === MvNormalMeanPrecision
    @test promote_variate_type(Multivariate, NormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision

    @test promote_variate_type(Univariate, MvNormalMeanCovariance) === NormalMeanVariance
    @test promote_variate_type(Univariate, MvNormalMeanPrecision) === NormalMeanPrecision
    @test promote_variate_type(Univariate, MvNormalWeightedMeanPrecision) === NormalWeightedMeanPrecision

    @test promote_variate_type(Multivariate, MvNormalMeanCovariance) === MvNormalMeanCovariance
    @test promote_variate_type(Multivariate, MvNormalMeanPrecision) === MvNormalMeanPrecision
    @test promote_variate_type(Multivariate, MvNormalWeightedMeanPrecision) === MvNormalWeightedMeanPrecision
end

@testitem "NormalFamily: Sampling univariate" begin
    include("./normal_family_setuptests.jl")

    rng = MersenneTwister(1234)

    for T in (Float32, Float64)
        let # NormalMeanVariance
            μ, v = 10randn(rng), 10rand(rng)
            d = convert(NormalMeanVariance{T}, μ, v)

            @test typeof(rand(d)) <: T

            samples = rand(rng, d, 5_000)

            @test isapprox(mean(samples), μ, atol = 0.5)
            @test isapprox(var(samples), v, atol = 0.5)
        end

        let # NormalMeanPrecision
            μ, w = 10randn(rng), 10rand(rng)
            d = convert(NormalMeanPrecision{T}, μ, w)

            @test typeof(rand(d)) <: T

            samples = rand(rng, d, 5_000)

            @test isapprox(mean(samples), μ, atol = 0.5)
            @test isapprox(inv(var(samples)), w, atol = 0.5)
        end

        let # WeightedMeanPrecision
            wμ, w = 10randn(rng), 10rand(rng)
            d = convert(NormalWeightedMeanPrecision{T}, wμ, w)

            @test typeof(rand(d)) <: T

            samples = rand(rng, d, 5_000)

            @test isapprox(inv(var(samples)) * mean(samples), wμ, atol = 0.5)
            @test isapprox(inv(var(samples)), w, atol = 0.5)
        end
    end
end

@testitem "NormalFamily: Sampling multivariate" begin
    include("./normal_family_setuptests.jl")

    rng = MersenneTwister(1234)
    for n in (2, 3), T in (Float64,), nsamples in (10_000,)
        μ = randn(rng, n)
        L = randn(rng, n, n)
        Σ = L * L'

        d = convert(MvNormalMeanCovariance{T}, μ, Σ)
        @test typeof(rand(d)) <: Vector{T}

        samples = eachcol(rand(rng, d, nsamples))
        weights = fill(1 / nsamples, nsamples)

        @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
        @test isapprox(
            sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
            cov(d),
            atol = n * 0.5
        )

        μ = randn(rng, n)
        L = randn(rng, n, n)
        W = L * L'
        d = convert(MvNormalMeanCovariance{T}, μ, W)
        @test typeof(rand(d)) <: Vector{T}

        samples = eachcol(rand(rng, d, nsamples))
        weights = fill(1 / nsamples, nsamples)

        @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
        @test isapprox(
            sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
            cov(d),
            atol = n * 0.5
        )

        ξ = randn(rng, n)
        L = randn(rng, n, n)
        W = L * L'

        d = convert(MvNormalWeightedMeanPrecision{T}, ξ, W)

        @test typeof(rand(d)) <: Vector{T}

        samples = eachcol(rand(rng, d, nsamples))
        weights = fill(1 / nsamples, nsamples)

        @test isapprox(sum(sample for sample in samples) / nsamples, mean(d), atol = n * 0.5)
        @test isapprox(
            sum((sample - mean(d)) * (sample - mean(d))' for sample in samples) / nsamples,
            cov(d),
            atol = n * 0.5
        )
    end
end

@testitem "NormalFamily: ExponentialFamilyDistribution{NormalMeanVariance}" begin
    include("./normal_family_setuptests.jl")

    for μ in -10.0:5.0:10.0, σ² in 0.1:1.0:5.0, T in union_types(UnivariateNormalDistributionsFamily)
        @testset let d = convert(T, NormalMeanVariance(μ, σ²))
            ef = test_exponentialfamily_interface(d)

            (η₁, η₂) = (mean(d) / var(d), -1 / 2var(d))

            for x in 10randn(4)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) ≈ 1 / sqrt(2π)
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x, abs2(x)))
                @test @inferred(logpartition(ef)) ≈ (-η₁^2 / 4η₂ - 1 / 2 * log(-2η₂))
                @test @inferred(insupport(ef, x))
            end
        end
    end

    # Test failing isproper cases
    @test !isproper(MeanParametersSpace(), NormalMeanVariance, [-1])
    @test !isproper(MeanParametersSpace(), NormalMeanVariance, [1, -0.1])
    @test !isproper(MeanParametersSpace(), NormalMeanVariance, [-0.1, -1])
    @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [-1.1])
    @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [1, 1])
    @test !isproper(NaturalParametersSpace(), NormalMeanVariance, [-1.1, 1])
end

@testitem "NormalFamily: prod with ExponentialFamilyDistribution{NormalMeanVariance}" begin
    include("./normal_family_setuptests.jl")

    for μleft in 10randn(4), σ²left in 10rand(4), μright in 10randn(4),
        σ²right in 10rand(4), Tleft in union_types(UnivariateNormalDistributionsFamily),
        Tright in union_types(UnivariateNormalDistributionsFamily)

        @testset let (left, right) = (convert(Tleft, NormalMeanVariance(μleft, σ²left)), convert(Tright, NormalMeanVariance(μright, σ²right)))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{NormalMeanVariance})
                )
            )
        end
    end
end

@testitem "NormalFamily: ExponentialFamilyDistribution{MvNormalMeanCovariance}" begin
    include("./normal_family_setuptests.jl")

    for s in (2, 3), T in union_types(MultivariateNormalDistributionsFamily)
        μ = 10randn(s)
        L = LowerTriangular(randn(s, s) + s * I)
        Σ = L * L'
        @testset let d = convert(T, MvNormalMeanCovariance(μ, Σ))
            ef = test_exponentialfamily_interface(
                d;
                # These are handled differently below
                test_fisherinformation_against_hessian = false,
                test_fisherinformation_against_jacobian = false,
                test_gradlogpartition_properties = false
            )

            (η₁, η₂) = (inv(Σ) * mean(d), -inv(Σ) / 2)

            for x in [10randn(s) for _ in 1:4]
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) ≈ (2π)^(-s / 2)
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x, x * x'))
                @test @inferred(logpartition(ef)) ≈ -1 / 4 * (η₁' * inv(η₂) * η₁) - 1 / 2 * logdet(-2η₂)
                @test @inferred(insupport(ef, x))
            end

            run_test_gradlogpartition_properties(d, test_against_forwardiff = false)

            # Extra test with AD-friendly logpartition function
            lp_ag = ForwardDiff.gradient(getlogpartitionfortest(NaturalParametersSpace(), MvNormalMeanCovariance), getnaturalparameters(ef))
            @test gradlogpartition(ef) ≈ lp_ag
        end
    end

    # Test failing isproper cases (naive)
    @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [-1])
    @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [1, -0.1])
    @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [-0.1, -1])
    @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [-1, 2, 3, 4]) # shapes are incompatible
    @test !isproper(MeanParametersSpace(), MvNormalMeanCovariance, [1, -0.1, -1, 0, 0, -1]) # covariance is not posdef

    @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [-1.1])
    @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [1, 1])
    @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [-1.1, 1])
    @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [-1, 2, 3, 4]) # shapes are incompatible
    @test !isproper(NaturalParametersSpace(), MvNormalMeanCovariance, [1, -0.1, 1, 0, 0, 1]) # -η₂ is not posdef
end

@testitem "NormalFamily: Fisher information matrix in natural parameters space" begin
    include("./normal_family_setuptests.jl")

    for i in 1:5, d in 2:10
        rng = StableRNG(d * i)
        μ = 10randn(rng, d)
        L = LowerTriangular(randn(rng, d, d) + d * I)
        Σ = L * L'
        ef = convert(ExponentialFamilyDistribution, MvNormalMeanCovariance(μ, Σ))

        fi_ef = fisherinformation(ef)
        # @test_broken isposdef(fi_ef)
        # The `isposdef` check is not really reliable in Julia, here, instead we compute eigen values
        @test issymmetric(fi_ef) || (LowerTriangular(fi_ef) ≈ (UpperTriangular(fi_ef)'))
        @test isposdef(fi_ef) || all(>(0), eigvals(fi_ef))

        fi_ef_inv = inv(fi_ef)
        @test (fi_ef_inv * fi_ef) ≈ Diagonal(ones(d + d^2))

        # WARNING: ForwardDiff returns a non-positive definite Hessian for a convex function. 
        # The matrices are identical up to permutations, resulting in eigenvalues that are the same up to a sign.
        fi_ag = ForwardDiff.hessian(getlogpartitionfortest(NaturalParametersSpace(), MvNormalMeanCovariance), getnaturalparameters(ef))
        @test norm(sort(eigvals(fi_ef)) - sort(abs.(eigvals(fi_ag)))) ≈ 0 atol = (1e-9 * d^2)
    end
end

# We normally perform test with jacobian transformation, but autograd fails to compute jacobians with duplicated elements.
@testitem "Fisher information matrix in mean parameters space" begin
    include("./normal_family_setuptests.jl")

    for i in 1:5, d in 2:3
        rng = StableRNG(d * i)
        μ = 10randn(rng, d)
        L = LowerTriangular(randn(rng, d, d) + d * I)
        Σ = L * L'
        n_samples = 10000
        dist = MvNormalMeanCovariance(μ, Σ)

        samples = rand(rng, dist, n_samples)

        θ = pack_parameters(MvNormalMeanCovariance, (μ, Σ))

        approxHessian = zeros(length(θ), length(θ))
        for sample in eachcol(samples)
            approxHessian -= ForwardDiff.hessian(Base.Fix2(gaussianlpdffortest, sample), θ)
        end
        approxFisherInformation = approxHessian /= n_samples

        # The error will be higher for sampling tests, tolerance adjusted accordingly.
        fi_dist = getfisherinformation(MeanParametersSpace(), MvNormalMeanCovariance)(θ)
        @test isposdef(fi_dist) || all(>(0), eigvals(fi_dist))
        @test issymmetric(fi_dist) || (LowerTriangular(fi_dist) ≈ (UpperTriangular(fi_dist)'))
        @test sort(eigvals(fi_dist)) ≈ sort(abs.(eigvals(approxFisherInformation))) rtol = 1e-1
        @test sort(svd(fi_dist).S) ≈ sort(svd(approxFisherInformation).S) rtol = 1e-1
    end
end

@testitem "Differentiability of ExponentialFamily(ExponentialFamily.MvNormalMeanCovariance) logpdf" begin
    include("./normal_family_setuptests.jl")
    for i in 1:5, d in 2:3
        rng = StableRNG(d * i)
        μ = 10randn(rng, d)
        L = LowerTriangular(randn(rng, d, d) + d * I)
        Σ = L * L'
        n_samples = 1
        dist = MvNormalMeanCovariance(μ, Σ)

        samples = rand(rng, dist, n_samples)

        θ = pack_parameters(MvNormalMeanCovariance, (μ, Σ))
        ef = convert(ExponentialFamilyDistribution, MvNormalMeanCovariance(μ, Σ))

        nat_space2mean_space = (η) -> begin
            dist = convert(Distribution, ExponentialFamilyDistribution(MvNormalMeanCovariance, η, nothing, nothing))
            μ, Σ = mean(dist), cov(dist)
            pack_parameters(MvNormalMeanCovariance, (μ, Σ))
        end

        for sample in eachcol(samples)
            mean_gradient = ForwardDiff.gradient(Base.Fix2(gaussianlpdffortest, sample), θ)
            nat_gradient = ForwardDiff.gradient(
                (η) -> logpdf(ExponentialFamilyDistribution(MvNormalMeanCovariance, η, nothing, nothing), sample),
                getnaturalparameters(ef)
            )
            jacobian = ForwardDiff.jacobian(nat_space2mean_space, getnaturalparameters(ef))
            # # autograd failing to compute jacobian of matrix part correclty. Comparing only vector (mean) part.
            @test nat_gradient[1:d] ≈ (jacobian'*mean_gradient)[1:d]
        end
    end
end

@testitem "MvNormalMeanCovariance: compute_logscale" begin
    include("./normal_family_setuptests.jl")

    cases = (
        (; μL = [1.0, -1.0], ΣL = [1.67 0.18; 0.18 1.97],
            μR = [2.5, 1.2], ΣR = [3.0 1.0; 1.0 2.0], expected = -3.9512453085506345),
        (; μL = [0.0, 0.0], ΣL = [1.0 0.0; 0.0 1.0],
            μR = [1.0, -1.0], ΣR = [2.0 0.5; 0.5 1.0], expected = -3.234216124248758),
        (; μL = [-2.0, 3.0], ΣL = [3.0 1.0; 1.0 2.0],
            μR = [0.5, 0.5], ΣR = [1.5 0.2; 0.2 1.5], expected = -5.439495426785419),
        (; μL = [10.0, -10.0], ΣL = [5.0 0.18; 0.18 2.0],
            μR = [2.5, 1.2], ΣR = [3.0 1.0; 1.0 2.0], expected = -26.855784396907417),
        (; μL = [0.0, 0.0, 0.0], ΣL = Matrix(Diagonal([1.0, 2.0, 3.0])),
            μR = [0.0, 0.0, 0.0], ΣR = Matrix(Diagonal([1.0, 2.0, 3.0])), expected = -4.692416105067964),
        (; μL = [1.0, 2.0, 3.0], ΣL = [2.0 0.1 0.0; 0.1 2.0 0.1; 0.0 0.1 2.0],
            μR = [-1.0, -2.0, -3.0], ΣR = [3.0 0.5 0.2; 0.5 3.0 0.5; 0.2 0.5 3.0], expected = -10.04009658793601),
        (; μL = [0.0, 1.0, 2.0, 3.0], ΣL = Matrix(Diagonal([1.0, 1.0, 1.0, 1.0])),
            μR = [1.0, 2.0, 3.0, 4.0], ΣR = Matrix(Diagonal([1.0, 2.0, 3.0, 4.0])), expected = -6.711166670876381)
    )

    # Parametrizations to test (skip known problematic conversion target)
    mv_types = [MvNormalMeanCovariance, MvNormalMeanPrecision, MvNormalWeightedMeanPrecision]

    for case in cases
        # Baseline expectation from canonical Float64 covariance form
        left64 = MvNormalMeanCovariance(case.μL, case.ΣL)
        right64 = MvNormalMeanCovariance(case.μR, case.ΣR)
        expected = case.expected

        @test compute_logscale(right64, right64, left64) ≈ expected

        for F in (Float64,)
            for T in mv_types
                leftT = convert(T{F}, left64)
                rightT = convert(T{F}, right64)
                originalL = convert(MvNormalMeanCovariance{Float64}, leftT)

                @test compute_logscale(leftT, leftT, rightT) ≈ expected rtol = 0.1
                @test compute_logscale(rightT, rightT, leftT) ≈ expected rtol = 0.1
            end
        end
    end
end
