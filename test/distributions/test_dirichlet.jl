module DirichletTest

using Test
using ExponentialFamily
using Distributions
using Random
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation
import SpecialFunctions: loggamma
using ForwardDiff
using LoopVectorization

include("../testutils.jl")

@testset "Dirichlet" begin

    # Dirichlet comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        @test_throws MethodError vague(Dirichlet)

        d1 = vague(Dirichlet, 2)

        @test typeof(d1) <: Dirichlet
        @test probvec(d1) == ones(2)

        d2 = vague(Dirichlet, 4)

        @test typeof(d2) <: Dirichlet
        @test probvec(d2) == ones(4)
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Dirichlet([1.0, 1.0, 1.0])) ≈ [-1.5000000000000002, -1.5000000000000002, -1.5000000000000002]
        @test mean(log, Dirichlet([1.1, 2.0, 2.0])) ≈ [-1.9517644694670657, -1.1052251939575213, -1.1052251939575213]
        @test mean(log, Dirichlet([3.0, 1.2, 5.0])) ≈ [-1.2410879175727905, -2.4529121492634465, -0.657754584239457]
    end

    @testset "ExponentialFamilyDistribution{Dirichlet}" begin
        @testset for len in 3:5
            α = rand(len)
            @testset let d = Dirichlet(α)
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false)
                η1 = getnaturalparameters(ef)

                for x in [ rand(len) for _ in 1:3 ]
                    x = x ./ sum(x)
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === 1.0
                    @test @inferred(sufficientstatistics(ef, x)) == (vmap(log,x),)
                    firstterm = mapreduce(x -> loggamma(x + 1), +, η1)
                    secondterm = loggamma(sum(η1) + length(η1))
                    @test @inferred(logpartition(ef)) ≈ firstterm - secondterm
                end
            end
        end

        for space in (MeanParametersSpace(), NaturalParametersSpace())
            @test !isproper(space, Dirichlet, [Inf, Inf], 1.0)
            @test !isproper(space, Dirichlet, [1.0], Inf)
            @test !isproper(space, Dirichlet, [NaN], 1.0)
            @test !isproper(space, Dirichlet, [1.0], NaN)
            @test !isproper(space, Dirichlet, [0.5, 0.5], 1.0)
            @test isproper(space, Dirichlet, [2.0, 3.0])
            @test !isproper(space, Dirichlet, [-1.0 ,-1.2])
        end

        @test_throws Exception convert(ExponentialFamilyDistribution, Dirichlet([Inf, Inf]))
    end

    @testset "prod with Distribution" begin
        @test default_prod_rule(Dirichlet, Dirichlet) === ClosedProd()

        @test @inferred(prod(ClosedProd(), Dirichlet([1.0, 1.0, 1.0]), Dirichlet([1.0, 1.0, 1.0]))) ≈ Dirichlet([1.0, 1.0, 1.0])
        @test @inferred(prod(ClosedProd(), Dirichlet([1.1, 1.0, 2.0]), Dirichlet([1.0, 1.2, 1.0]))) ≈ Dirichlet([1.1, 1.2000000000000002, 2.0])
        @test @inferred(prod(ClosedProd(), Dirichlet([1.1, 2.0, 2.0]), Dirichlet([3.0, 1.2, 5.0]))) ≈ Dirichlet([3.0999999999999996, 2.2, 6.0])

        # GenericProd should always check the default strategy and fallback if available
        @test @inferred(prod(GenericProd(), Dirichlet([1.0, 1.0, 1.0]), Dirichlet([1.0, 1.0, 1.0]))) ≈ Dirichlet([1.0, 1.0, 1.0])
        @test @inferred(prod(GenericProd(), Dirichlet([1.1, 1.0, 2.0]), Dirichlet([1.0, 1.2, 1.0]))) ≈ Dirichlet([1.1, 1.2000000000000002, 2.0])
        @test @inferred(prod(GenericProd(), Dirichlet([1.1, 2.0, 2.0]), Dirichlet([3.0, 1.2, 5.0]))) ≈ Dirichlet([3.0999999999999996, 2.2, 6.0])
    end

    @testset "prod with ExponentialFamilyDistribution" for len=3:6
        αleft = rand(len) .+ 1
        αright = rand(len) .+ 1
        let left = Dirichlet(αleft), right = Dirichlet(αright)
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

end
