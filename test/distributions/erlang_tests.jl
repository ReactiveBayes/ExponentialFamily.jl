
# Erlang comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Erlang: vague" begin
    include("distributions_setuptests.jl")

    @test Erlang() == Erlang(1, 1.0)
    @test vague(Erlang) == Erlang(1, 1e12)
end

@testitem "Erlang: mean(::typeof(log))" begin
    include("distributions_setuptests.jl")

    @test mean(log, Erlang(1, 3.0)) ≈ digamma(1) + log(3.0)
    @test mean(log, Erlang(2, 0.3)) ≈ digamma(2) + log(0.3)
    @test mean(log, Erlang(3, 0.3)) ≈ digamma(3) + log(0.3)
end

@testitem "Erlang: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for a in 1:3, b in 1.0:1.0:3.0
        @testset let d = Erlang(a, b)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)

            (η1, η2) = (a - 1, -inv(b))
            for x in 10rand(4)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === oneunit(x)
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (log(x), x))
                @test @inferred(logpartition(ef)) ≈ (logfactorial(η1) - (η1 + one(η1)) * log(-η2))
            end

            @test !@inferred(insupport(ef, -0.5))
            @test @inferred(insupport(ef, 0.5))

            # Not in the support
            @test_throws Exception logpdf(ef, -0.5)
        end
    end

    # Test failing isproper cases
    @test !isproper(MeanParametersSpace(), Erlang, [-1])
    @test !isproper(MeanParametersSpace(), Erlang, [1, -0.1])
    @test !isproper(MeanParametersSpace(), Erlang, [-0.1, 1])
    @test !isproper(NaturalParametersSpace(), Erlang, [-1.1])
    @test isproper(NaturalParametersSpace(), Erlang, [1, -1.1])
    @test !isproper(NaturalParametersSpace(), Erlang, [-1.1, 1])
end

@testitem "Erlang: prod with Distributions" begin
    include("distributions_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
        @test prod(strategy, Erlang(1, 1), Erlang(1, 1)) == Erlang(1, 1 / 2)
        @test prod(strategy, Erlang(1, 2), Erlang(1, 1)) == Erlang(1, 2 / 3)
        @test prod(strategy, Erlang(1, 2), Erlang(1, 2)) == Erlang(1, 1)
        @test prod(strategy, Erlang(2, 2), Erlang(1, 2)) == Erlang(2, 1)
        @test prod(strategy, Erlang(2, 2), Erlang(2, 2)) == Erlang(3, 1)
    end

    @test @allocated(prod(ClosedProd(), Erlang(1, 1), Erlang(1, 1))) == 0
end

@testitem "Erlang: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for aleft in 1:3, aright in 2:5, bleft in 0.51:1.0:5.0, bright in 0.51:1.0:5.0
        @testset let (left, right) = (Erlang(aleft, bleft), Erlang(aright, bright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{Erlang})
                )
            )
        end
    end
end
