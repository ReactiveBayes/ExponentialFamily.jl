
# Rayleigh comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Rayleigh: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for σ in 10rand(4)
        @testset let d = Rayleigh(σ)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)
            η1 = first(getnaturalparameters(ef))

            for x in 10rand(4)
                @test @inferred(isbasemeasureconstant(ef)) === NonConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === x
                @test @inferred(sufficientstatistics(ef, x)) === (x^2,)
                @test @inferred(logpartition(ef)) ≈ -log(-2 * η1)
            end
        end
    end

    for space in (MeanParametersSpace(), NaturalParametersSpace())
        @test !isproper(space, Rayleigh, [Inf])
        @test !isproper(space, Rayleigh, [NaN])
        @test !isproper(space, Rayleigh, [1.0], NaN)
        @test !isproper(space, Rayleigh, [0.5, 0.5], 1.0)
    end
    @test !isproper(MeanParametersSpace(), Rayleigh, [-1.0])
    @test_throws Exception convert(ExponentialFamilyDistribution, Rayleigh(Inf))
end

@testitem "Rayleigh: prod with PreserveTypeProd{ExponentialFamilyDistribution}" begin
    include("distributions_setuptests.jl")

    for σleft in 1:4, σright in 4:7
        @testset let (left, right) = (Rayleigh(σleft), Rayleigh(σright))
            ef_left = convert(ExponentialFamilyDistribution, left)
            ef_right = convert(ExponentialFamilyDistribution, right)
            prod_dist = prod(PreserveTypeProd(ExponentialFamilyDistribution), left, right)
            @test first(hquadrature(x -> pdf(prod_dist, tan(x * pi / 2)) * (pi / 2) * (1 / cos(x * pi / 2)^2), 0.0, 1.0)) ≈ 1.0
            @test getnaturalparameters(prod_dist) == getnaturalparameters(ef_left) + getnaturalparameters(ef_right)
        end
    end
end
