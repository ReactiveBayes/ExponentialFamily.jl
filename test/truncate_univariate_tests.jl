# --- List of univariate exponential-family distributions ---
    cont_dists = [
        Normal(0.0, 1.0),            # Continuous
        Normal(2.0, 3.0),            # Continuous
        Gamma(2.0, 2.0),             # Continuous, rate parameterization
        Weibull(1.5, 1.0)            # Continuous
    ]

    

@testitem "TruncateExponentialFamily" begin
    discrete_dists = [
        Poisson(3),                  # Discrete
        Bernoulli(0.7),              # Discrete
        Binomial(10, 0.5),           # Discrete
    ]
    for d in discrete_dists
        d_ef =  convert(ExponentialFamilyDistribution, d)
        # Compute reasonable bounds for truncation
        
        # --- Define test bound configurations ---
        bound_configs_discrete = [
            (0.0, 5.0),
            (1.0, 3.0),
            (2, 10),
            (0, 0),
            (1, 8.0)
        ]

        bound_configs_continuous = [
            (-2.0, 2.0),
            (-1.0, 1.0),
            (0.0, 3.0),
            (-Inf, 0.0),
            (0.5, Inf)
        ]

        for (l, u) in bound_configs_discrete
            d_trunc = TruncatedExponentialFamilyDistribution(d_ef, l, u)
            # Test bounds
            @test d_trunc.lower <= d_trunc.upper
            #@test d_trunc.lcdf ≈ cdf(d_trunc.untruncated, d_trunc.lower)
            #@test d_trunc.ucdf ≈ cdf(d_trunc.untruncated, d_trunc.upper)
        end

    end
end

