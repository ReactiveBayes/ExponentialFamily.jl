
@testitem "GammaFamily: Base statistical methods" begin
    include("./gamma_family_setuptests.jl")

    types = union_types(GammaDistributionsFamily{Float64})
    rng   = MersenneTwister(1234)
    for _ in 1:10
        for type in types
            left = convert(type, 100 * rand(rng, Float64), 100 * rand(rng, Float64))
            for type in types
                right = convert(type, left)
                @test compare_basic_statistics(left, right)

                @test all(params(DefaultParametersSpace(), left) .== (shape(left), scale(left)))
                @test all(params(DefaultParametersSpace(), right) .== (shape(right), scale(right)))
            end
        end
    end
end

@testitem "GammaFamily: ExponentialFamilyDistribution" begin
    include("./gamma_family_setuptests.jl")

    for k in (0.1, 2.0, 5.0), θ in (0.1, 2.0, 5.0), T in union_types(GammaDistributionsFamily{Float64})
        @testset let d = convert(T, GammaShapeScale(k, θ))
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true, test_gradlogpartition_default_space = true)

            (η₁, η₂) = (shape(d) - 1, -inv(scale(d)))

            for x in 0.1:0.5:5.0
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === oneunit(x)
                @test all(@inferred(sufficientstatistics(ef, x)) .=== (log(x), x))
                @test @inferred(logpartition(ef)) ≈ loggamma(η₁ + 1) - (η₁ + 1) * log(-η₂)
            end

            @test @inferred(insupport(ef, 0.5))
            @test !@inferred(insupport(ef, -0.5))

            # # Not in the support
            @test_throws Exception logpdf(ef, -0.5)
        end
    end

    # Test failing isproper cases
    @test !isproper(DefaultParametersSpace(), Gamma, [-1])
    @test !isproper(DefaultParametersSpace(), Gamma, [1, -1])
    @test !isproper(DefaultParametersSpace(), Gamma, [-1, -1])
    @test !isproper(NaturalParametersSpace(), Gamma, [-1])
    @test !isproper(NaturalParametersSpace(), Gamma, [1, 10])
    @test !isproper(NaturalParametersSpace(), Gamma, [-100, -1])

    # shapes must add up to something more than 1, otherwise is not proper
    let ef = convert(ExponentialFamilyDistribution, Gamma(0.1, 1.0))
        @test !isproper(prod(PreserveTypeProd(ExponentialFamilyDistribution), ef, ef))
    end
end

@testitem "GammaFamily: prod with ExponentialFamilyDistribution" begin
    include("./gamma_family_setuptests.jl")

    for kleft in 0.51:1.0:5.0, kright in 0.51:1.0:5.0, θleft in 0.1:1.0:5.0, θright in 0.1:1.0:5.0, Tleft in union_types(GammaDistributionsFamily{Float64}),
        Tright in union_types(GammaDistributionsFamily{Float64})

        @testset let (left, right) = (convert(Tleft, Gamma(kleft, θleft)), convert(Tright, Gamma(kright, θright)))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{Gamma})
                )
            )
        end
    end
end

@testitem "GammaFamily: mean(::typeof(inv))" begin
    include("./gamma_family_setuptests.jl")

    for α in 2.0:1.0:5.0, β in 0.1:1.0:5.0, T in union_types(GammaDistributionsFamily{Float64})
        @testset let d = convert(T, GammaShapeRate(α, β))
            invgamma = InverseGamma(α, β)
            @test shape(invgamma.invd) == shape(d)
            @test scale(invgamma.invd) == scale(d)
            @test mean(inv, d) ≈ mean(invgamma)
        end
    end

    # Test that the mean of the inverse is Inf when the shape is less than 1
    @test mean(inv, GammaShapeScale(0.5, 1.0)) == Inf
end

@testitem "GammaFamily: mean(::typeof(loggamma))" begin
    include("./gamma_family_setuptests.jl")

    import QuadGK: quadgk

    # (shape, scale) => E[log Γ(x)], computed by a 256-bit quadrature of `log Γ(x)` against the density
    references = [
        (0.01, 1.0) => 100.56014626169652,
        (0.3, 2.0) => 3.0284013650363115,
        (1.0, 1.0) => 0.843302107531978,
        (2.0, 0.5) => 0.41564637083354294,
        (10.0, 0.1) => 0.08236877619855938,
        (5.0, 100.0) => 2653.507430149158,
        (0.5, 1e-4) => 11.173821543382113,
        (1e3, 1e-3) => 0.0008224774237053027,
        (1e6, 1.0) => 1.2815505069147779e7,
        (1e6, 1e-6) => 8.22467043821409e-7
    ]

    for ((k, θ), expected) in references, T in union_types(GammaDistributionsFamily{Float64})
        @testset let d = convert(T, GammaShapeScale(k, θ))
            @test mean(loggamma, d) ≈ expected rtol = 1e-13 atol = 1e-15
        end
    end

    # An independent check by quadrature of `log Γ(x)` itself
    for k in (0.7, 3.0, 40.0), θ in (0.05, 1.0, 20.0)
        d = GammaShapeScale(k, θ)
        m, sd = mean(d), std(d)
        points = filter(>(0), [m - 6sd, m - 2sd, m, m + 2sd, m + 6sd])
        expected, _ = quadgk(x -> pdf(d, x) * loggamma(x), 0.0, points..., Inf; rtol = 1e-12)
        @test mean(loggamma, d) ≈ expected rtol = 1e-9
    end

    # Differentiable in both parameters
    for (a, b) in ((2.0, 0.5), (0.01, 3.0), (1e5, 1e-3))
        f(a, b) = mean(loggamma, GammaShapeRate(a, b))
        h = 1e-6
        @test ForwardDiff.derivative(a -> f(a, b), a) ≈ (f(a * (1 + h), b) - f(a * (1 - h), b)) / (2h * a) rtol = 1e-6
        @test ForwardDiff.derivative(b -> f(a, b), b) ≈ (f(a, b * (1 + h)) - f(a, b * (1 - h))) / (2h * b) rtol = 1e-6
    end

    @test mean(loggamma, GammaShapeRate(2.0f0, 2.0f0)) isa Float32
    @test mean(loggamma, GammaShapeRate(2.0f0, 2.0f0)) ≈ 0.41564637083354294 rtol = 1e-5
end
