# Bernoulli comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "Bernoulli: vague" begin
    include("distributions_setuptests.jl")

    d = vague(Bernoulli)

    @test typeof(d) <: Bernoulli
    @test mean(d) === 0.5
    @test succprob(d) === 0.5
    @test failprob(d) === 0.5
end

@testitem "Bernoulli: probvec" begin
    include("distributions_setuptests.jl")

    @test probvec(Bernoulli(0.5)) === (0.5, 0.5)
    @test probvec(Bernoulli(0.3)) === (0.7, 0.3)
    @test probvec(Bernoulli(0.6)) === (0.4, 0.6)
end

@testitem "Bernoulli: logscale Bernoulli-Bernoulli/Categorical" begin
    include("distributions_setuptests.jl")

    @test compute_logscale(Bernoulli(0.5), Bernoulli(0.5), Bernoulli(0.5)) ≈ log(0.5)
    @test compute_logscale(Bernoulli(1), Bernoulli(0.5), Bernoulli(1)) ≈ log(0.5)
    @test compute_logscale(Categorical([0.5, 0.5]), Bernoulli(0.5), Categorical([0.5, 0.5])) ≈ log(0.5)
    @test compute_logscale(Categorical([0.5, 0.5]), Categorical([0.5, 0.5]), Bernoulli(0.5)) ≈ log(0.5)
    @test compute_logscale(Categorical([1.0, 0.0]), Bernoulli(0.5), Categorical([1])) ≈ log(0.5)
    @test compute_logscale(Categorical([1.0, 0.0, 0.0]), Bernoulli(0.5), Categorical([1.0, 0, 0])) ≈ log(0.5)
end

@testitem "Bernoulli: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for p in 0.1:0.1:0.9
        @testset let d = Bernoulli(p)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = true)
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

@testitem "Bernoulli: prod with Distribution" begin
    include("distributions_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
        @test @inferred(prod(strategy, Bernoulli(0.5), Bernoulli(0.5))) ≈ Bernoulli(0.5)
        @test @inferred(prod(strategy, Bernoulli(0.1), Bernoulli(0.6))) ≈ Bernoulli(0.14285714285714285)
        @test @inferred(prod(strategy, Bernoulli(0.78), Bernoulli(0.05))) ≈ Bernoulli(0.1572580645161291)
    end

    for strategy in (ClosedProd(), PreserveTypeProd(Distribution), GenericProd())
        # Test symmetric case
        @test @inferred(prod(strategy, Bernoulli(0.5), Categorical([0.5, 0.5]))) ≈ Categorical([0.5, 0.5])
        @test @inferred(prod(strategy, Categorical([0.5, 0.5]), Bernoulli(0.5))) ≈ Categorical([0.5, 0.5])
    end

    @test @allocated(prod(ClosedProd(), Bernoulli(0.5), Bernoulli(0.5))) === 0
    @test @allocated(prod(PreserveTypeProd(Distribution), Bernoulli(0.5), Bernoulli(0.5))) === 0
    @test @allocated(prod(GenericProd(), Bernoulli(0.5), Bernoulli(0.5))) === 0
end

@testitem "Bernoulli: prod with Categorical" begin
    include("distributions_setuptests.jl")

    @test prod(ClosedProd(), Bernoulli(0.5), Categorical([0.5, 0.5])) ≈
          Categorical([0.5, 0.5])
    @test prod(ClosedProd(), Bernoulli(0.1), Categorical(0.4, 0.6)) ≈
          Categorical([1 - 0.14285714285714285, 0.14285714285714285])
    @test prod(ClosedProd(), Bernoulli(0.78), Categorical([0.95, 0.05])) ≈
          Categorical([1 - 0.1572580645161291, 0.1572580645161291])
    @test prod(ClosedProd(), Bernoulli(0.5), Categorical([0.3, 0.3, 0.4])) ≈
          Categorical([0.5, 0.5, 0])
    @test prod(ClosedProd(), Bernoulli(0.5), Categorical([1.0])) ≈
          Categorical([1.0, 0])
end

@testitem "Bernoulli: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for pleft in 0.1:0.1:0.9, pright in 0.1:0.1:0.9
        @testset let (left, right) = (Bernoulli(pleft), Bernoulli(pright))
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
