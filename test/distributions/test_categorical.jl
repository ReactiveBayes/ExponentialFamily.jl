module CategoricalTest

using ExponentialFamily, Distributions
using Test, ForwardDiff, Random, StatsFuns, StableRNGs, LinearAlgebra

include("../testutils.jl")

@testset "Categorical" begin

    # Categorical comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        @test_throws MethodError vague(Categorical)

        d1 = vague(Categorical, 2)

        @test typeof(d1) <: Categorical
        @test probvec(d1) ≈ [0.5, 0.5]

        d2 = vague(Categorical, 4)

        @test typeof(d2) <: Categorical
        @test probvec(d2) ≈ [0.25, 0.25, 0.25, 0.25]
    end

    @testset "probvec" begin
        @test probvec(Categorical([0.1, 0.4, 0.5])) == [0.1, 0.4, 0.5]
        @test probvec(Categorical([1 / 3, 1 / 3, 1 / 3])) == [1 / 3, 1 / 3, 1 / 3]
        @test probvec(Categorical([0.8, 0.1, 0.1])) == [0.8, 0.1, 0.1]
    end

    @testset "prod Distribution" begin
        @test prod(ClosedProd(), Categorical([0.1, 0.4, 0.5]), Categorical([1 / 3, 1 / 3, 1 / 3])) ==
              Categorical([0.1, 0.4, 0.5])
        @test prod(ClosedProd(), Categorical([0.1, 0.4, 0.5]), Categorical([0.8, 0.1, 0.1])) ==
              Categorical([0.47058823529411764, 0.23529411764705882, 0.2941176470588235])
        @test prod(ClosedProd(), Categorical([0.2, 0.6, 0.2]), Categorical([0.8, 0.1, 0.1])) ≈
              Categorical([2 / 3, 1 / 4, 1 / 12])
    end

    @testset "ExponentialFamilyDistribution{Categorical}" begin
        @testset for s in (2, 3, 4, 5)
            @testset let d = Categorical(normalize!(rand(s), 1))
                ef = test_exponentialfamily_interface(
                    d;
                    test_distribution_conversion = false,
                    test_basic_functions = false,
                    test_fisherinformation_against_hessian = false,
                    test_fisherinformation_against_jacobian = false
                )

                run_test_distribution_conversion(d; assume_no_allocations = false)
                run_test_basic_functions(d; assume_no_allocations = false)
                run_test_fisherinformation_against_hessian(d; assume_no_allocations = false)
                run_test_fisherinformation_against_jacobian(d; assume_no_allocations = false, mappings = (
                    NaturalParametersSpace() => MeanParametersSpace(),
                    # MeanParametersSpace() => NaturalParametersSpace(), # here is the problem for discussion, the test is broken
                ))

                # (η₁, η₂) = (a - 1, b - 1)

                # for x in 0.1:0.1:0.9
                #     @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                #     @test @inferred(basemeasure(ef, x)) === oneunit(x)
                #     @test all(@inferred(sufficientstatistics(ef, x)) .≈ (log(x), log(1 - x)))
                #     @test @inferred(logpartition(ef)) ≈ (logbeta(η₁ + 1, η₂ + 1))
                # end

                # @test !@inferred(insupport(ef, -0.5))
                # @test @inferred(insupport(ef, 0.5))

                # # Not in the support
                # @test_throws Exception logpdf(ef, -0.5)
            end
        end

        # Test failing isproper cases
        # @test !isproper(MeanParametersSpace(), Beta, [-1])
        # @test !isproper(MeanParametersSpace(), Beta, [1, -0.1])
        # @test !isproper(MeanParametersSpace(), Beta, [-0.1, 1])
        # @test !isproper(NaturalParametersSpace(), Beta, [-1.1])
        # @test !isproper(NaturalParametersSpace(), Beta, [1, -1.1])
        # @test !isproper(NaturalParametersSpace(), Beta, [-1.1, 1])

    end

    # @testset "prod ExponentialFamilyDistribution" begin
    #     for d in 2:20
    #         pleft     = rand(d)
    #         pleft     = pleft ./ sum(pleft)
    #         distleft  = Categorical(pleft)
    #         efleft    = convert(ExponentialFamilyDistribution, distleft)
    #         ηleft     = getnaturalparameters(efleft)
    #         pright    = rand(d)
    #         pright    = pright ./ sum(pright)
    #         distright = Categorical(pright)
    #         efright   = convert(ExponentialFamilyDistribution, distright)
    #         ηright    = getnaturalparameters(efright)
    #         efprod    = prod(efleft, efright)
    #         distprod  = prod(ClosedProd(), distleft, distright)
    #         @test efprod == ExponentialFamilyDistribution(Categorical, ηleft + ηright)
    #         @test convert(Distribution, efprod) ≈ distprod
    #     end
    # end

    # @testset "fisher information" begin
    #     function transformation(η)
    #         expη = exp.(η)
    #         expη / sum(expη)
    #     end

    #     rng = StableRNG(42)
    #     for n in 2:5
    #         p = rand(rng, Dirichlet(ones(n)))
    #         dist = Categorical(p)
    #         ef = convert(ExponentialFamilyDistribution, dist)
    #         η = getnaturalparameters(ef)

    #         f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Categorical, η))
    #         autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
    #         @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-10

    #         J = ForwardDiff.jacobian(transformation, η)
    #         @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) atol = 1e-10
    #     end
    # end

    # @testset "ExponentialFamilyDistribution mean var" begin
    #     rng = StableRNG(42)
    #     for n in 2:10
    #         p = rand(rng, Dirichlet(ones(n)))
    #         dist = Categorical(p)
    #         ef = convert(ExponentialFamilyDistribution, dist)
    #         @test mean(dist) ≈ mean(ef) atol = 1e-8
    #         @test var(dist) ≈ var(ef) atol = 1e-8
    #     end
    # end
end

end
