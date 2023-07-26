module CategoricalTest

using Test
using ExponentialFamily
using Distributions
using StableRNGs
using Random
using ForwardDiff
import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, basemeasure, fisherinformation, sufficientstatistics
import LogExpFunctions: logsumexp

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

    @testset "natural parameters related" begin
        ηcat = ExponentialFamilyDistribution(Categorical, log.([1, 1]))
        dist = Categorical([1/2, 1/2])
        η1   = ExponentialFamilyDistribution(Categorical, log.([1, 1]))
        η2   = ExponentialFamilyDistribution(Categorical, log.([1, 1]))

        @test convert(Distribution, ηcat) == dist
        @test convert(ExponentialFamilyDistribution, dist) ==
              ExponentialFamilyDistribution(Categorical, log.([1, 1]))
        @test prod(η1, η2) == ExponentialFamilyDistribution(Categorical, [0.0, 0.0])

        @test logpdf(ηcat, 2) == logpdf(dist, 2)
        @test logpdf(ηcat, 1) == logpdf(dist, 1)
        @test_throws AssertionError logpdf(ηcat, 0.5) == logpdf(dist, 0.5)

        @test basemeasure(η1, 2) == 1.0
        @test basemeasure(η1, 1) == 1.0
        @test_throws AssertionError basemeasure(η1, rand()) == 1.0
        @test_throws AssertionError basemeasure(η2, rand()) == 1.0

        @test logpartition(ηcat) == logsumexp(getnaturalparameters(ηcat))

        categoricalef = ExponentialFamilyDistribution(Categorical, [0.1, 0.2, 0.3, 0.4])
        @test sufficientstatistics(categoricalef, 3) == [0, 0, 1, 0]
        @test_throws AssertionError sufficientstatistics(categoricalef, 5) == [0, 0, 0, 0]
        @test_throws AssertionError sufficientstatistics(categoricalef, [0, 1]) == [0, 1, 0, 0]
        @test sufficientstatistics(categoricalef, 4) == [0, 0, 0, 1]
    end

    @testset "prod Distribution" begin
        @test prod(ClosedProd(), Categorical([0.1, 0.4, 0.5]), Categorical([1 / 3, 1 / 3, 1 / 3])) ==
              Categorical([0.1, 0.4, 0.5])
        @test prod(ClosedProd(), Categorical([0.1, 0.4, 0.5]), Categorical([0.8, 0.1, 0.1])) ==
              Categorical([0.47058823529411764, 0.23529411764705882, 0.2941176470588235])
        @test prod(ClosedProd(), Categorical([0.2, 0.6, 0.2]), Categorical([0.8, 0.1, 0.1])) ≈
              Categorical([2 / 3, 1 / 4, 1 / 12])
    end

    @testset "prod ExponentialFamilyDistribution" begin
        for d in 2:20
            pleft     = rand(d)
            pleft     = pleft ./ sum(pleft)
            distleft  = Categorical(pleft)
            efleft    = convert(ExponentialFamilyDistribution, distleft)
            ηleft     = getnaturalparameters(efleft)
            pright    = rand(d)
            pright    = pright ./ sum(pright)
            distright = Categorical(pright)
            efright   = convert(ExponentialFamilyDistribution, distright)
            ηright    = getnaturalparameters(efright)
            efprod    = prod(efleft, efright)
            distprod  = prod(ClosedProd(), distleft, distright)
            @test efprod == ExponentialFamilyDistribution(Categorical, ηleft + ηright)
            @test convert(Distribution, efprod) ≈ distprod
        end
    end

    @testset "fisher information" begin
        function transformation(η)
            expη = exp.(η)
            expη / sum(expη)
        end

        rng = StableRNG(42)
        for n in 2:5
            p = rand(rng, Dirichlet(ones(n)))
            dist = Categorical(p)
            ef = convert(ExponentialFamilyDistribution, dist)
            η = getnaturalparameters(ef)

            f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Categorical, η))
            autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
            @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-10

            J = ForwardDiff.jacobian(transformation, η)
            @test J' * fisherinformation(dist) * J ≈ fisherinformation(ef) atol = 1e-10
        end
    end

    @testset "ExponentialFamilyDistribution mean var" begin
        rng = StableRNG(42)
        for n in 2:10
            p = rand(rng, Dirichlet(ones(n)))
            dist = Categorical(p)
            ef = convert(ExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
            @test var(dist) ≈ var(ef) atol = 1e-8
        end
    end
end

end
