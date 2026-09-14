
# Dirichlet comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Dirichlet: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(Dirichlet)

    d1 = vague(Dirichlet, 2)
    @test typeof(d1) <: Dirichlet
end

@testitem "Dirichlet: mean(::typeof(log))" begin
    include("distributions_setuptests.jl")

    import Base.Broadcast: BroadcastFunction

    @test mean(BroadcastFunction(log), Dirichlet([1.0, 1.0, 1.0])) ≈ [-1.5000000000000002, -1.5000000000000002, -1.5000000000000002]
    @test mean(BroadcastFunction(log), Dirichlet([1.1, 2.0, 2.0])) ≈ [-1.9517644694670657, -1.1052251939575213, -1.1052251939575213]
    @test mean(BroadcastFunction(log), Dirichlet([3.0, 1.2, 5.0])) ≈ [-1.2410879175727905, -2.4529121492634465, -0.657754584239457]
end

@testitem "Dirichlet: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    rng = StableRNG(42)
    for len in 3:5
        α = rand(rng, len)
        @testset let d = Dirichlet(α)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false)
            η1 = getnaturalparameters(ef)

            for x in [rand(rng, len) for _ in 1:3]
                x = x ./ sum(x)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === 1.0
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (map(log, x),))
                firstterm = mapreduce(x -> loggamma(x + 1), +, η1)
                secondterm = loggamma(sum(η1) + length(η1))
                @test @inferred(logpartition(ef)) ≈ firstterm - secondterm
            end
        end
    end

    for space in (DefaultParametersSpace(), NaturalParametersSpace())
        @test !isproper(space, Dirichlet, [Inf, Inf], 1.0)
        @test !isproper(space, Dirichlet, [1.0], Inf)
        @test !isproper(space, Dirichlet, [NaN], 1.0)
        @test !isproper(space, Dirichlet, [1.0], NaN)
        @test !isproper(space, Dirichlet, [0.5, 0.5], 1.0)
        @test isproper(space, Dirichlet, [2.0, 3.0])
        @test !isproper(space, Dirichlet, [-1.0, -1.2])
        @test !isproper(space, Dirichlet, [NaN, 1.0, 1.0])
    end

    @test_throws Exception convert(ExponentialFamilyDistribution, Dirichlet([Inf, Inf]))
end

@testitem "Dirichlet: prod with Distribution" begin
    include("distributions_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
        @test @inferred(prod(strategy, Dirichlet([1.0, 1.0, 1.0]), Dirichlet([1.0, 1.0, 1.0]))) ≈ Dirichlet([1.0, 1.0, 1.0])
        @test @inferred(prod(strategy, Dirichlet([1.1, 1.0, 2.0]), Dirichlet([1.0, 1.2, 1.0]))) ≈ Dirichlet([1.1, 1.2, 2.0])
        @test @inferred(prod(strategy, Dirichlet([1.1, 2.0, 2.0]), Dirichlet([3.0, 1.2, 5.0]))) ≈ Dirichlet([3.1, 2.2, 6.0])
    end
end

@testitem "Dirichlet: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")
    rng = StableRNG(123)
    for len in 3:6
        αleft = rand(rng, len) .+ 1
        αright = rand(rng, len) .+ 1
        @testset let (left, right) = (Dirichlet(αleft), Dirichlet(αright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd()
                )
            )
        end
    end
end

@testitem "Dirichlet: probvec does not exist" begin
    include("distributions_setuptests.jl")
    @test_throws ArgumentError probvec(Dirichlet([1.0, 2.0, 3.0]))
end
