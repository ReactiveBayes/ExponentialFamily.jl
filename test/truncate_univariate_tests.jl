@testitem "TruncateExponentialFamily" begin
    import BayesBase: params, rand
    import Distributions: islowerbounded, isupperbounded
    import Random
    # Compute reasonable bounds for truncation

    # --- Define test bound configurations ---
    valid_bound_configs = [
        (0.0, 5.0),
        (1.0, 3.0),
        (2, 10),
        (0, 1.0),
        (1, 8.0),
        (-2.0, 2.0),
        (-1.0, 1.0),
        (0.0, 3.0),
        (-Inf, 2.0),
        (nothing, 1.0),
        (0.5, Inf),
        (0.5, nothing),
        (-Inf64, Inf64),
        (-Inf32, Inf64)
    ]

    dists = [
        Poisson(3),                  # Discrete
        Binomial(10, 0.5),           # Discrete
        #NormalMeanVariance(0.0, 1.0),            # Continuous
        #NormalMeanVariance(2.0, 3.0),            # Continuous
        Gamma(2.0, 2.0),             # Continuous, rate parameterization
        Weibull(1.5, 1.0)            # Continuous
    ]
    #test valid modes of behavior
    for d in dists
        d_ef = convert(ExponentialFamilyDistribution, d)

        for (l, u) in valid_bound_configs
            d_trunc = TruncatedExponentialFamilyDistribution(d_ef, l, u)
            # Test bounds
            @test d_trunc.lower <= d_trunc.upper
            @test d_trunc.lower >= something(l, -Inf)
            @test d_trunc.upper <= something(u, Inf)
            @test islowerbounded(d_trunc) == true

            #Test functionality
            @test minimum(d_trunc) >= minimum(d)
            @test maximum(d_trunc) <= maximum(d)
            @test insupport(d_trunc, -50.0) == false

            #Test parameters
            new_params = params(d_trunc)
            old_params = params(d);
            @test all(isapprox.(new_params[1:(end-2)], old_params[1:end])) #converting back and forth introduces some small deviation 

            #Test sampling mechanism
            rng = Random.MersenneTwister(42)
            randomNumber = rand(rng, d_trunc)
            @test randomNumber isa Real && randomNumber <= d_trunc.upper && randomNumber >= d_trunc.lower
        end
    end
end

# Regression test for issue #290: TruncatedExponentialFamilyDistribution previously
# defined no logpdf/pdf/cdf, so `logpdf(dtr, x)` threw a MethodError. Compare the new
# density/cdf against Distributions.truncated (ground truth) for continuous cases.
@testitem "TruncateExponentialFamily: logpdf/pdf/cdf" begin
    import BayesBase: logpdf, pdf
    import Distributions: cdf, truncated

    # Continuous: match Distributions.truncated exactly (P(X = boundary) = 0)
    for (d, l, u) in ((Gamma(2.0, 2.0), 1.0, 5.0), (Weibull(1.5, 1.0), 0.5, 3.0))
        d_ef = convert(ExponentialFamilyDistribution, d)
        d_trunc = TruncatedExponentialFamilyDistribution(d_ef, l, u)
        ref = truncated(d, l, u)

        for x in range(l + 1e-3, u - 1e-3; length = 7)
            @test logpdf(d_trunc, x) ≈ logpdf(ref, x)
            @test pdf(d_trunc, x) ≈ pdf(ref, x)
            @test cdf(d_trunc, x) ≈ cdf(ref, x)
        end

        # outside the truncated support
        @test logpdf(d_trunc, l - 1.0) == -Inf
        @test pdf(d_trunc, u + 1.0) == 0.0
        @test cdf(d_trunc, l - 1.0) == 0.0
        @test cdf(d_trunc, u + 1.0) == 1.0
    end

    # Discrete: the methods now exist (previously MethodError) and are self-consistent
    d_trunc = TruncatedExponentialFamilyDistribution(convert(ExponentialFamilyDistribution, Poisson(3)), 1.0, 8.0)
    @test hasmethod(logpdf, Tuple{typeof(d_trunc), Float64})
    for x in 1:8
        @test isfinite(logpdf(d_trunc, x))
        @test pdf(d_trunc, x) ≈ exp(logpdf(d_trunc, x))
    end
    @test logpdf(d_trunc, 0) == -Inf
    @test 0.0 <= cdf(d_trunc, 4) <= 1.0
end
