module MatrixDirichletTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff
import SpecialFunctions: loggamma
import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure,fisherinformation, as_vec, logpartition, reconstructargument!

function reconstructed_logpartition(ef::KnownExponentialFamilyDistribution{T}, ηvec) where {T}
    natural_params = getnaturalparameters(ef)
    mean_size = first(size(natural_params))
    matrix = reshape(ηvec, mean_size, mean_size)
    ef = KnownExponentialFamilyDistribution(T, Matrix(matrix))
    return logpartition(ef)
end

function test_partition(ef::KnownExponentialFamilyDistribution{MatrixDirichlet})
    η =  getnaturalparameters(ef)
    return sum(loggamma.(η.+1.0)) - sum(loggamma.(sum(η.+1.0, dims=1)))
end
@testset "MatrixDirichlet" begin
    @testset "common" begin
        @test MatrixDirichlet <: Distribution
        @test MatrixDirichlet <: ContinuousDistribution
        @test MatrixDirichlet <: MatrixDistribution

        @test value_support(MatrixDirichlet) === Continuous
        @test variate_form(MatrixDirichlet) === Matrixvariate
    end

    @testset "vague" begin
        @test_throws MethodError vague(MatrixDirichlet)

        d1 = vague(MatrixDirichlet, 3)

        @test typeof(d1) <: MatrixDirichlet
        @test mean(d1) == ones(3, 3) ./ sum(ones(3, 3); dims = 1)

        d2 = vague(MatrixDirichlet, 4)

        @test typeof(d2) <: MatrixDirichlet
        @test mean(d2) == ones(4, 4) ./ sum(ones(4, 4); dims = 1)

        @test vague(MatrixDirichlet, 3, 3) == vague(MatrixDirichlet, (3, 3))
        @test vague(MatrixDirichlet, 4, 4) == vague(MatrixDirichlet, (4, 4))
        @test vague(MatrixDirichlet, 3, 4) == vague(MatrixDirichlet, (3, 4))
        @test vague(MatrixDirichlet, 4, 3) == vague(MatrixDirichlet, (4, 3))

        d3 = vague(MatrixDirichlet, 3, 4)

        @test typeof(d3) <: MatrixDirichlet
        @test mean(d3) == ones(3, 4) ./ sum(ones(3, 4); dims = 1)
    end

    @testset "entropy" begin
        @test entropy(MatrixDirichlet([1.0 1.0; 1.0 1.0; 1.0 1.0])) ≈ -1.3862943611198906
        @test entropy(MatrixDirichlet([1.2 3.3; 4.0 5.0; 2.0 1.1])) ≈ -3.1139933152617787
        @test entropy(MatrixDirichlet([0.2 3.4; 5.0 11.0; 0.2 0.6])) ≈ -11.444984495104693
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, MatrixDirichlet([1.0 1.0; 1.0 1.0; 1.0 1.0])) ≈ [
            -1.5000000000000002 -1.5000000000000002
            -1.5000000000000002 -1.5000000000000002
            -1.5000000000000002 -1.5000000000000002
        ]
        @test mean(log, MatrixDirichlet([1.2 3.3; 4.0 5.0; 2.0 1.1])) ≈ [
            -2.1920720408623637 -1.1517536610071326
            -0.646914475838374 -0.680458481634953
            -1.480247809171707 -2.6103310904778305
        ]
        @test mean(log, MatrixDirichlet([0.2 3.4; 5.0 11.0; 0.2 0.6])) ≈ [
            -6.879998107291004 -1.604778825293528
            -0.08484054226701443 -0.32259407259407213
            -6.879998107291004 -4.214965875553984
        ]
    end

    @testset "prod" begin
        d1 = MatrixDirichlet([0.2 3.4; 5.0 11.0; 0.2 0.6])
        d2 = MatrixDirichlet([1.2 3.3; 4.0 5.0; 2.0 1.1])
        d3 = MatrixDirichlet([1.0 1.0; 1.0 1.0; 1.0 1.0])

        @test prod(ClosedProd(), d1, d2) ==
              MatrixDirichlet([0.3999999999999999 5.699999999999999; 8.0 15.0; 1.2000000000000002 0.7000000000000002])
        @test prod(ClosedProd(), d1, d3) == MatrixDirichlet(
            [0.19999999999999996 3.4000000000000004; 5.0 11.0; 0.19999999999999996 0.6000000000000001]
        )
        @test prod(ClosedProd(), d2, d3) == MatrixDirichlet([1.2000000000000002 3.3; 4.0 5.0; 2.0 1.1])
    end

    @testset "promote_variate_type" begin
        @test_throws MethodError promote_variate_type(Univariate, MatrixDirichlet)

        @test promote_variate_type(Multivariate, Dirichlet) === Dirichlet
        @test promote_variate_type(Matrixvariate, Dirichlet) === MatrixDirichlet

        @test promote_variate_type(Multivariate, MatrixDirichlet) === Dirichlet
        @test promote_variate_type(Matrixvariate, MatrixDirichlet) === MatrixDirichlet
    end

    @testset "natural parameters related" begin
        @test convert(KnownExponentialFamilyDistribution, MatrixDirichlet([0.6 0.7; 1.0 2.0])) ==
              KnownExponentialFamilyDistribution(MatrixDirichlet, [0.6 0.7; 1.0 2.0] .- 1)
        b_01 = MatrixDirichlet([1.0 10.0; 2.0 10.0])
        nb_01 = convert(KnownExponentialFamilyDistribution, b_01)
        @test logpartition(nb_01) ==
              mapreduce(
            d -> logpartition(KnownExponentialFamilyDistribution(Dirichlet, d)),
            +,
            eachcol(getnaturalparameters(nb_01))
        )
        for i in 1:9
            b = MatrixDirichlet([i/10.0 i/20; i/5 i])
            bnp = convert(KnownExponentialFamilyDistribution, b)
            @test convert(Distribution, bnp) ≈ b
            @test logpdf(bnp, [0.5 0.4; 0.2 0.3]) ≈ logpdf(b, [0.5 0.4; 0.2 0.3])
            @test logpdf(bnp, [0.5 0.4; 0.2 0.3]) ≈ logpdf(b, [0.5 0.4; 0.2 0.3])

            @test convert(KnownExponentialFamilyDistribution, b) == bnp

            @test prod(nb_01, bnp) ≈ convert(KnownExponentialFamilyDistribution, prod(ClosedProd(), b_01, b))
            @test logpartition(bnp) ≈ test_partition(bnp)
        end
        @test isproper(KnownExponentialFamilyDistribution(MatrixDirichlet, [10 2; 3 2])) === true
    end

    @testset "KnownExponentialFamilyDistribution mean" begin
        for i in 1:9
            dist = MatrixDirichlet([i/10.0 i/20; i/5 i])
            ef = convert(KnownExponentialFamilyDistribution, dist)
            @test mean(dist) ≈ mean(ef) atol = 1e-8
        end
    end

    @testset "fisher information" begin
        for i in 1:9
            dist = MatrixDirichlet([i/10.0 i/20; i/5 i])
            ef = convert(KnownExponentialFamilyDistribution, dist)
            η = vcat(as_vec(getnaturalparameters(ef)))
            f_logpartition = (η_vec) -> reconstructed_logpartition(ef, η_vec)
         
            @test fisherinformation(ef) ≈ ForwardDiff.hessian(f_logpartition, η) rtol = 1e-8
            @test fisherinformation(dist) ≈ fisherinformation(ef) atol = 1e-8 ##Jacobian is omitted because it is identity
        end
    end
end

end
