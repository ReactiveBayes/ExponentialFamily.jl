module DirichletTest

using Test
using ExponentialFamily
using Distributions
using Random
import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure
import SpecialFunctions: loggamma
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
end

@testset "probvec" begin
    @test probvec(Dirichlet([1.0, 1.0, 1.0])) == [1.0, 1.0, 1.0]
    @test probvec(Dirichlet([1.1, 2.0, 2.0])) == [1.1, 2.0, 2.0]
    @test probvec(Dirichlet([3.0, 1.2, 5.0])) == [3.0, 1.2, 5.0]
end

@testset "mean(::typeof(log))" begin
    @test mean(log, Dirichlet([1.0, 1.0, 1.0])) ≈ [-1.5000000000000002, -1.5000000000000002, -1.5000000000000002]
    @test mean(log, Dirichlet([1.1, 2.0, 2.0])) ≈ [-1.9517644694670657, -1.1052251939575213, -1.1052251939575213]
    @test mean(log, Dirichlet([3.0, 1.2, 5.0])) ≈ [-1.2410879175727905, -2.4529121492634465, -0.657754584239457]
end

@testset "promote_variate_type" begin
    @test_throws MethodError promote_variate_type(Univariate, Dirichlet)

    @test promote_variate_type(Multivariate, Dirichlet) === Dirichlet
    @test promote_variate_type(Matrixvariate, Dirichlet) === MatrixDirichlet

    @test promote_variate_type(Multivariate, MatrixDirichlet) === Dirichlet
    @test promote_variate_type(Matrixvariate, MatrixDirichlet) === MatrixDirichlet
end

@testset "natural parameters related" begin
    @test convert(KnownExponentialFamilyDistribution, Dirichlet([0.6, 0.7])) ==
          KnownExponentialFamilyDistribution(Dirichlet, [0.6, 0.7] .- 1)

    @test logpartition(convert(KnownExponentialFamilyDistribution, Dirichlet([1, 1]))) ≈ 2loggamma(2)
    @test logpartition(convert(KnownExponentialFamilyDistribution, Dirichlet([0.1, 0.2]))) ≈
          loggamma(0.1) + loggamma(0.2) - loggamma(0.3)
    @test isproper(KnownExponentialFamilyDistribution(Dirichlet, [10, 2, 3])) === true
    @test isproper(KnownExponentialFamilyDistribution(Dirichlet, [-0.1, -0.2, 3])) === true
    @test isproper(KnownExponentialFamilyDistribution(Dirichlet, [-0.1, -0.2, -3])) === false
    @test_throws AssertionError basemeasure(KnownExponentialFamilyDistribution(Dirichlet, [-0.1, -0.2, -3]), rand()) == 1.0
    for i in 1:9
        b = Dirichlet([i / 10.0, i / 5, i])
        bnp = convert(KnownExponentialFamilyDistribution, b)
        @test convert(Distribution, bnp) ≈ b
        @test logpdf(bnp, [0.5, 0.4, 0.1]) ≈ logpdf(b, [0.5, 0.4, 0.1])
        @test logpdf(bnp, [0.2, 0.3, 0.5]) ≈ logpdf(b, [0.2, 0.3, 0.5])
    end
end

@testset "prod" begin
    @test prod(ClosedProd(), Dirichlet([1.0, 1.0, 1.0]), Dirichlet([1.0, 1.0, 1.0])) ==
          Dirichlet([1.0, 1.0, 1.0])
    @test prod(ClosedProd(), Dirichlet([1.1, 1.0, 2.0]), Dirichlet([1.0, 1.2, 1.0])) ==
          Dirichlet([1.1, 1.2000000000000002, 2.0])
    @test prod(ClosedProd(), Dirichlet([1.1, 2.0, 2.0]), Dirichlet([3.0, 1.2, 5.0])) ==
          Dirichlet([3.0999999999999996, 2.2, 6.0])

    b_01 = Dirichlet([10.0, 10.0, 10.0])
    nb_01 = convert(KnownExponentialFamilyDistribution, b_01)
    for i in 1:9
        b = Dirichlet([i / 10.0, i / 5, i])
        bnp = convert(KnownExponentialFamilyDistribution, b)
        @test prod(ClosedProd(), b, b_01) ≈ convert(Distribution, prod(bnp, nb_01))
    end
end

end
