module BetaTest

using Test
using ExponentialFamily
using Distributions
using Random
using ForwardDiff

import ExponentialFamily: mirrorlog, KnownExponentialFamilyDistribution, getnaturalparameters, logpartition,
    basemeasure, sufficientstatistics, fisherinformation
import SpecialFunctions: loggamma

@testset "Beta" begin

    # Beta comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(Beta)

        @test typeof(d) <: Beta
        @test mean(d) === 0.5
        @test params(d) === (1.0, 1.0)
    end

    @testset "mean(::typeof(log))" begin
        @test mean(log, Beta(1.0, 3.0)) ≈ -1.8333333333333335
        @test mean(log, Beta(0.1, 0.3)) ≈ -7.862370395825961
        @test mean(log, Beta(4.5, 0.3)) ≈ -0.07197681436958758
    end

    @testset "mean(::typeof(mirrorlog))" begin
        @test mean(mirrorlog, Beta(1.0, 3.0)) ≈ -0.33333333333333337
        @test mean(mirrorlog, Beta(0.1, 0.3)) ≈ -0.9411396776150167
        @test mean(mirrorlog, Beta(4.5, 0.3)) ≈ -4.963371962929249
    end

    @testset "prod" begin
        @test prod(ClosedProd(), Beta(3.0, 2.0), Beta(2.0, 1.0)) ≈ Beta(4.0, 2.0)
        @test prod(ClosedProd(), Beta(7.0, 1.0), Beta(0.1, 4.5)) ≈ Beta(6.1, 4.5)
        @test prod(ClosedProd(), Beta(1.0, 3.0), Beta(0.2, 0.4)) ≈ Beta(0.19999999999999996, 2.4)
    end

    @testset "natural parameters related" begin
        betaef = KnownExponentialFamilyDistribution(Beta, [1, 0.2])
        @test sufficientstatistics(betaef, 0.1) == [log(0.1), log(1.0 - 0.1)]
        @test sufficientstatistics(betaef, 0.9) == [log(0.9), log(1.0 - 0.9)]
        @test sufficientstatistics(betaef, 0.999) == [log(0.999), log(1.0 - 0.999)]
        @test_throws AssertionError sufficientstatistics(betaef, 1.01)
        @test_throws AssertionError sufficientstatistics(betaef, -0.01)

        for i in 0:10, j in 0:10
            @test convert(Distribution, KnownExponentialFamilyDistribution(Beta, [i, j])) == Beta(i + 1, j + 1)
            @test convert(KnownExponentialFamilyDistribution, Beta(i + 1, j + 1)) ==
                  KnownExponentialFamilyDistribution(Beta, [i, j])
        end

        @test logpartition(KnownExponentialFamilyDistribution(Beta, [0, 0])) ≈ 0
        @test logpartition(KnownExponentialFamilyDistribution(Beta, [1, 1])) ≈ -loggamma(4)

        for i in 0:10, j in 0:10
            @test logpdf(KnownExponentialFamilyDistribution(Beta, [i, j]), 0.01) ≈ logpdf(Beta(i + 1, j + 1), 0.01)
            @test logpdf(KnownExponentialFamilyDistribution(Beta, [i, j]), 0.5) ≈ logpdf(Beta(i + 1, j + 1), 0.5)
        end

        for i in 0:10
            @test isproper(KnownExponentialFamilyDistribution(Beta, [i, i])) === true
        end
        for i in 1:10
            @test isproper(KnownExponentialFamilyDistribution(Beta, [-i, -i])) === false
        end

        for i in 1:10, j in 1:10
            @test basemeasure(KnownExponentialFamilyDistribution(Beta, [i, j]), rand()) == 1.0
        end

        @testset "prod with KnownExponentialFamilyDistribution" begin
            for αleft in 0.01:1:50, βleft in 0.01:1:10
                left   = Beta(αleft, βleft)
                efleft = convert(KnownExponentialFamilyDistribution, left)
                ηleft  = getnaturalparameters(efleft)
                for αright in 0.01:1:50, βright in 0.01:1:10
                    right   = Beta(αright, βright)
                    efright = convert(KnownExponentialFamilyDistribution, right)
                    ηright  = getnaturalparameters(efright)
                    @test prod(ClosedProd(), efleft, efright) ==
                          KnownExponentialFamilyDistribution(Beta, ηleft + ηright)
                    @test prod(efleft, efright) == KnownExponentialFamilyDistribution(Beta, ηleft + ηright)
                    if isless(αleft + αright - 1, 0) || isless(βleft + βright - 1, 0)
                        @test_throws DomainError prod(ClosedProd(), left, right) == convert(
                            Distribution,
                            KnownExponentialFamilyDistribution(Beta, ηleft + ηright)
                        )
                    else
                        @test prod(ClosedProd(), left, right) ≈
                              convert(Distribution, KnownExponentialFamilyDistribution(Beta, ηleft + ηright))
                    end
                end
            end
        end
        @testset "fisherinformation" begin
            for a in 0.01:1:10
                for b in 0.01:1:10
                    dist = Beta(a, b)
                    ef = convert(KnownExponentialFamilyDistribution, dist)
                    η = getnaturalparameters(ef)

                    f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(Beta, η))
                    autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
                    @test fisherinformation(ef) ≈ autograd_information(η) atol = 1e-10
                    # Here Jacobian is identity matrix. To speed up the tests its computation is omitted
                    @test fisherinformation(dist) ≈ fisherinformation(ef) atol = 1e-10
                end
            end
        end
    end
end

end
