
# Laplace comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Laplace: vague" begin
    include("distributions_setuptests.jl")

    d = vague(Laplace)

    @test typeof(d) <: Laplace
    @test mean(d) === 0.0
    @test params(d) === (0.0, 1e12)
end

@testitem "Laplace: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for location in (-1.0, 0.0, 1.0), scale in (0.25, 0.5, 2.0)
        @testset let d = Laplace(location, scale)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)
            η₁ = -1 / scale

            for x in (-1.0, 0.0, 1.0)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === oneunit(x)
                @test @inferred(sufficientstatistics(ef, x)) === (abs(x - location),)
                @test @inferred(logpartition(ef)) ≈ log(-2 / η₁)
            end
        end
    end

    for space in (MeanParametersSpace(), NaturalParametersSpace())
        @test !isproper(space, Laplace, [Inf], 1.0)
        @test !isproper(space, Laplace, [1.0], Inf)
        @test !isproper(space, Laplace, [NaN], 1.0)
        @test !isproper(space, Laplace, [1.0], NaN)
        @test !isproper(space, Laplace, [0.5, 0.5], 1.0)

        # Conditioner is required
        @test_throws Exception isproper(space, Laplace, [0.5], [0.5, 0.5])
        @test_throws Exception isproper(space, Laplace, [1.0], nothing)
        @test_throws Exception isproper(space, Laplace, [1.0], nothing)
    end

    @test_throws Exception convert(ExponentialFamilyDistribution, Laplace(Inf, Inf))
end

@testitem "Laplace: prod with Distribution" begin
    include("distributions_setuptests.jl")

    @test default_prod_rule(Laplace, Laplace) === PreserveTypeProd(Laplace)

    @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(0.0, 0.5), Laplace(0.0, 0.5))) ≈ Laplace(0.0, 0.25)
    @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(1.0, 1.0), Laplace(1.0, 1.0))) ≈ Laplace(1.0, 0.5)
    @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(2.0, 3.0), Laplace(2.0, 7.0))) ≈ Laplace(2.0, 2.1)

    # GenericProd should always check the default strategy and fallback if available
    @test @inferred(prod(GenericProd(), Laplace(0.0, 0.5), Laplace(0.0, 0.5))) ≈ Laplace(0.0, 0.25)
    @test @inferred(prod(GenericProd(), Laplace(1.0, 1.0), Laplace(1.0, 1.0))) ≈ Laplace(1.0, 0.5)
    @test @inferred(prod(GenericProd(), Laplace(2.0, 3.0), Laplace(2.0, 7.0))) ≈ Laplace(2.0, 2.1)

    # Different location parameters cannot be compute a closed prod with the same type
    @test_throws Exception prod(PreserveTypeProd(Laplace), Laplace(0.0, 0.5), Laplace(0.01, 0.5))
    @test_throws Exception prod(PreserveTypeProd(Laplace), Laplace(1.0, 0.5), Laplace(-1.0, 0.5))
end

@testitem "Laplace: prod with ExponentialFamilyDistribution: same location parameter" begin
    include("distributions_setuptests.jl")

    for location in (0.0, 1.0), sleft in 0.1:0.1:0.9, sright in 0.1:0.1:0.9
        @testset let (left, right) = (Laplace(location, sleft), Laplace(location, sright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (PreserveTypeProd(ExponentialFamilyDistribution{Laplace}), GenericProd())
            )
        end
    end

    # Different location parameters cannot be compute a closed prod with the same type
    @test_throws Exception prod(
        PreserveTypeProd(ExponentialFamilyDistribution{Laplace}),
        convert(ExponentialFamilyDistribution, Laplace(0.0, 0.5)),
        convert(ExponentialFamilyDistribution, Laplace(0.01, 0.5))
    )
    @test_throws Exception prod(
        PreserveTypeProd(ExponentialFamilyDistribution{Laplace}),
        convert(ExponentialFamilyDistribution, Laplace(1.0, 0.5)),
        convert(ExponentialFamilyDistribution, Laplace(-1.0, 0.5))
    )
end

@testitem "Laplace: prod with ExponentialFamilyDistribution: different location parameter" begin
    include("distributions_setuptests.jl")

    for locationleft in (0.0, 1.0), sleft in 0.1:0.1:0.4, locationright in (2.0, 3.0), sright in 1.1:0.1:1.3
        @testset let (left, right) = (Laplace(locationleft, sleft), Laplace(locationright, sright))
            ef_left = convert(ExponentialFamilyDistribution, left)
            ef_right = convert(ExponentialFamilyDistribution, right)
            ef_prod = prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_left, ef_right)
            @test first(hquadrature(x -> pdf(ef_prod, tan(x * pi / 2)) * (pi / 2) * (1 / cos(x * pi / 2)^2), -1.0, 1.0)) ≈ 1.0 atol = 1e-6
        end
    end
end
