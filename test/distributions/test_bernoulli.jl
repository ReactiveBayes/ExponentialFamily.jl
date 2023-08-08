module BernoulliTest

using ExponentialFamily, Distributions
using Test, ForwardDiff, Random, StatsFuns, StableRNGs

include("../testutils.jl")

# Fisher information can in principle be computed with the `hessian` from `ForwardDiff` with relatively high-mean_precision
# Its fine to use it in tests, but we also check that our implementation is faster
fisherinformation_fortests(ef) = ForwardDiff.hessian(η -> getlogpartition(NaturalParametersSpace(), Bernoulli)(η), getnaturalparameters(ef))

@testset "Bernoulli" begin

    # Bernoulli comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(Bernoulli)

        @test typeof(d) <: Bernoulli
        @test mean(d) === 0.5
        @test succprob(d) === 0.5
        @test failprob(d) === 0.5
    end

    @testset "probvec" begin
        @test probvec(Bernoulli(0.5)) === (0.5, 0.5)
        @test probvec(Bernoulli(0.3)) === (0.7, 0.3)
        @test probvec(Bernoulli(0.6)) === (0.4, 0.6)
    end

    @testset "logscale Bernoulli-Bernoulli/Categorical" begin
        @test compute_logscale(Bernoulli(0.5), Bernoulli(0.5), Bernoulli(0.5)) ≈ log(0.5)
        @test compute_logscale(Bernoulli(1), Bernoulli(0.5), Bernoulli(1)) ≈ log(0.5)
        @test compute_logscale(Categorical([0.5, 0.5]), Bernoulli(0.5), Categorical([0.5, 0.5])) ≈ log(0.5)
        @test compute_logscale(Categorical([0.5, 0.5]), Categorical([0.5, 0.5]), Bernoulli(0.5)) ≈ log(0.5)
        @test compute_logscale(Categorical([1.0, 0.0]), Bernoulli(0.5), Categorical([1])) ≈ log(0.5)
        @test compute_logscale(Categorical([1.0, 0.0, 0.0]), Bernoulli(0.5), Categorical([1.0, 0, 0])) ≈ log(0.5)
    end

    @testset "ExponentialFamilyDistribution{Bernoulli}" begin
        @testset for p in 0.1:0.1:0.9
            @testset let d = Bernoulli(p)
                ef = test_exponentialfamily_interface(d)
                η₁ = logit(p)

                for x in (0, 1)
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) === oneunit(x)
                    @test @inferred(sufficientstatistics(ef, x)) === (x,)
                    @test @inferred(logpartition(ef)) ≈ log(1 + exp(η₁))
                end

                @test !@inferred(insupport(ef, -0.5))
                @test !@inferred(insupport(ef, 0.5))

                # Not in the support
                @test_throws Exception logpdf(ef, 0.5)
                @test_throws Exception logpdf(ef, -0.5)
            end
        end

        # Test failing isproper cases
        @test !isproper(MeanParametersSpace(), Bernoulli, [-1])
        @test !isproper(MeanParametersSpace(), Bernoulli, [0.5, 0.5])
        @test !isproper(NaturalParametersSpace(), Bernoulli, [0.5, 0.5])
        @test !isproper(NaturalParametersSpace(), Bernoulli, [Inf])

        @test_throws Exception convert(ExponentialFamilyDistribution, Bernoulli(1.0)) # We cannot convert from `1.0`, `logit` function returns `Inf`
    end

    @testset "prod with Distribution" begin
        @test default_prod_rule(Bernoulli, Bernoulli) === ClosedProd()

        @test @inferred(prod(ClosedProd(), Bernoulli(0.5), Bernoulli(0.5))) ≈ Bernoulli(0.5)
        @test @inferred(prod(ClosedProd(), Bernoulli(0.1), Bernoulli(0.6))) ≈ Bernoulli(0.14285714285714285)
        @test @inferred(prod(ClosedProd(), Bernoulli(0.78), Bernoulli(0.05))) ≈ Bernoulli(0.1572580645161291)

        # GenericProd should always check the default strategy and fallback if available
        @test @inferred(prod(GenericProd(), Bernoulli(0.5), Bernoulli(0.5))) ≈ Bernoulli(0.5)
        @test @inferred(prod(GenericProd(), Bernoulli(0.1), Bernoulli(0.6))) ≈ Bernoulli(0.14285714285714285)
        @test @inferred(prod(GenericProd(), Bernoulli(0.78), Bernoulli(0.05))) ≈ Bernoulli(0.1572580645161291)

        @test @allocated(prod(ClosedProd(), Bernoulli(0.5), Bernoulli(0.5))) === 0
        @test @allocated(prod(GenericProd(), Bernoulli(0.5), Bernoulli(0.5))) === 0
    end

    @testset "prod with ExponentialFamilyDistribution" for pleft in 0.1:0.1:0.9, pright in 0.1:0.1:0.9
        let left = Bernoulli(pleft), right = Bernoulli(pright)
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{Bernoulli})
                )
            )
        end
    end
end

end
