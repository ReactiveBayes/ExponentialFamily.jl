module ErlangTest

using Test
using ExponentialFamily
using Random
using Distributions
using ForwardDiff
import SpecialFunctions: logfactorial, loggamma,digamma
import ExponentialFamily:
    xtlog, ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation

include("../testutils.jl")

@testset "Erlang" begin
    @testset "Constructors" begin
        @test Erlang() == Erlang(1, 1.0)
        @test vague(Erlang) == Erlang(1, 1e12)
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Erlang(1, 3.0)) ≈ digamma(1) + log(3.0)
        @test mean(log, Erlang(2, 0.3)) ≈ digamma(2) + log(0.3)
        @test mean(log, Erlang(3, 0.3)) ≈ digamma(3) + log(0.3)
    end

    @testset "ExponentialFamilyDistribution{Erlang}" begin
        @testset for a in 1:3, b in 1.0:1.0:3.0
            @testset let d = Erlang(a, b)
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)

                (η1, η2) = (a - 1, -inv(b) )
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


    @testset "prod with Distributions" begin
        @test prod(ClosedProd(), Erlang(1, 1), Erlang(1, 1)) == Erlang(1, 1 / 2)
        @test prod(ClosedProd(), Erlang(1, 2), Erlang(1, 1)) == Erlang(1, 2 / 3)
        @test prod(ClosedProd(), Erlang(1, 2), Erlang(1, 2)) == Erlang(1, 1)
        @test prod(ClosedProd(), Erlang(2, 2), Erlang(1, 2)) == Erlang(2, 1)
        @test prod(ClosedProd(), Erlang(2, 2), Erlang(2, 2)) == Erlang(3, 1)

        @test @allocated(prod(ClosedProd(), Erlang(1, 1), Erlang(1, 1))) == 0
    end

    @testset "prod with ExponentialFamilyDistribution" for aleft in 1:3, aright in 2:5, bleft in 0.51:1.0:5.0, bright in 0.51:1.0:5.0
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

end
