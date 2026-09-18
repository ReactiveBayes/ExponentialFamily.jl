
# Weibull comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Weibull: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for shape in (1.0, 2.0, 3.0), scale in (0.25, 0.5, 2.0)
        @testset let d = Weibull(shape, scale)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false, test_fisherinformation_against_jacobian = false)
            η1 = first(getnaturalparameters(ef))
            run_test_fisherinformation_against_jacobian(
                d;
                assume_no_allocations = true,
                mappings = (
                    DefaultParametersSpace() => NaturalParametersSpace()
                )
            )
            for x in scale:1.0:(scale+3.0)
                @test @inferred(isbasemeasureconstant(ef)) === NonConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === x^(shape - 1)
                @test @inferred(sufficientstatistics(ef, x)) === (x^shape,)
                @test @inferred(logpartition(ef)) ≈ -log(-η1) - log(shape)
            end
        end
    end

    @testset "fisher information by natural to mean jacobian" begin
        @testset for k in (1, 3), λ in (0.1, 4.0)
            η = -(1 / λ)^k
            transformation(η) = [k, (-1 / η[1])^(1 / k)]
            J = ForwardDiff.jacobian(transformation, [η])
            @test first(J' * getfisherinformation(DefaultParametersSpace(), Weibull, k)(λ) * J) ≈
                  first(getfisherinformation(NaturalParametersSpace(), Weibull, k)(η)) atol = 1e-8
        end
    end

    for space in (DefaultParametersSpace(), NaturalParametersSpace())
        @test !isproper(space, Weibull, [Inf], 1.0)
        @test !isproper(space, Weibull, [1.0], Inf)
        @test !isproper(space, Weibull, [NaN], 1.0)
        @test !isproper(space, Weibull, [1.0], NaN)
        @test !isproper(space, Weibull, [0.5, 0.5], 1.0)

        # Conditioner is required
        @test_throws Exception isproper(space, Weibull, [0.5], [0.5, 0.5])
        @test_throws Exception isproper(space, Weibull, [1.0], nothing)
        @test_throws Exception isproper(space, Weibull, [1.0], nothing)
    end

    @test_throws Exception convert(ExponentialFamilyDistribution, Weibull(Inf, Inf))
end

@testitem "Weibull: prod with PreserveTypeProd{ExponentialFamilyDistribution} for the same conditioner" begin
    include("distributions_setuptests.jl")

    for η in -2.0:0.5:-0.5, k in 1.0:0.5:2, x in 0.5:0.5:2.0
        ef_left = convert(Distribution, ExponentialFamilyDistribution(Weibull, [η], k))
        ef_right = convert(Distribution, ExponentialFamilyDistribution(Weibull, [-η^2], k))
        res = prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_left, ef_right)
        @test getbasemeasure(res)(x) == x^(2 * (k - 1))
        @test sufficientstatistics(res, x) == (x^k,)
        @test getlogpartition(res)(η - η^2) ==
              log(abs(η - η^2)^(1 / k)) + loggamma(2 - 1 / k) - 2 * log(abs(η - η^2)) - log(k)
        @test getnaturalparameters(res) ≈ [η - η^2]
        @test first(hquadrature(x -> pdf(res, tan(x * pi / 2)) * (pi / 2) * (1 / cos(pi * x / 2))^2, 0.0, 1.0)) ≈
              1.0
    end
end

@testitem "Weibull: prod with PreserveTypeProd{ExponentialFamilyDistribution} for different k" begin
    include("distributions_setuptests.jl")

    for η in -12:4:-0.5, k in 1.0:4:10, x in 0.5:4:10
        ef_left = convert(Distribution, ExponentialFamilyDistribution(Weibull, [η], k * 2))
        ef_right = convert(Distribution, ExponentialFamilyDistribution(Weibull, [-η^2], k))
        res = prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_left, ef_right)
        @test getbasemeasure(res)(x) == x^(k + k * 2 - 2)
        @test sufficientstatistics(res, x) == (x^(2 * k), x^k)
        @test getnaturalparameters(res) ≈ [η, -η^2]
        @test first(hquadrature(x -> pdf(res, tan(x * pi / 2)) * (pi / 2) * (1 / cos(pi * x / 2))^2, 0.0, 1.0)) ≈
              1.0
    end
end

# Regression test for issue #292: the DefaultParametersSpace log-partition returned a
# 1-element SVector with the wrong value (SA[k/λ]) instead of the scalar k·log λ − log k,
# and the gradient referenced an undefined `η1` (UndefVarError).
@testitem "Weibull: default-space helpers consistency" begin
    include("distributions_setuptests.jl")

    for k in (1.5, 2.0, 3.0), λ in (0.5, 1.5, 2.5)
        θ = [λ]
        η_tup = MeanToNatural(Weibull)((λ,), k)
        η = pack_parameters(NaturalParametersSpace(), Weibull, η_tup)

        lp_def = getlogpartition(DefaultParametersSpace(), Weibull, k)(θ)
        lp_nat = getlogpartition(NaturalParametersSpace(), Weibull, k)(η)
        @test lp_def isa Real
        @test lp_def ≈ lp_nat
        @test lp_def ≈ k * log(λ) - log(k)

        g = getgradlogpartition(DefaultParametersSpace(), Weibull, k)(θ)
        @test g ≈ ForwardDiff.gradient(getlogpartition(DefaultParametersSpace(), Weibull, k), θ)
        @test g ≈ [k / λ]
    end
end
