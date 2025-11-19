@testitem "TruncateExponentialFamily" begin
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

    invalid_bound_configs = [
        (5.0, 2.0),
        (3.0, 1),
        (Inf, -Inf),
        (Inf32, -Inf64)
    ]

    dists = [
        Poisson(3),                  # Discrete
        #Bernoulli(0.7),              # Discrete
        #Binomial(10, 0.5),           # Discrete
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
            @test d_trunc.lower >= (something(l, -Inf))
            @test d_trunc.upper <= float(something(u, Inf))
            
            # Test parameters
            #println(typeof(d_trunc))
            #new_params = params(d_trunc)
            #@test new_params(1) == params(d_trunc.untruncated)
        end
    end
end
