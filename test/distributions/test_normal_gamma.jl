module NormalGammaTests

using ExponentialFamily, Distributions, Test, LinearAlgebra, StableRNGs, Random
using StatsFuns: loggamma

import ExponentialFamily: location

include("../testutils.jl")

@testset "NormalGamma" begin
    @testset "common" begin
        m = rand()
        s, a, b = 1.0, 0.1, 3.0
        dist = NormalGamma(m, s, a, b)
        @test params(dist) == (m, s, a, b)
        @test location(dist) == m
        @test scale(dist) == s
        @test shape(dist) == a
        @test rate(dist) == b
    end

    @testset "ExponentialFamilyDistribution{NormalGamma}" begin
        @testset for μ in 10randn(3), λ in 10rand(3), α in 10rand(3), β in 10 * rand(3)
            @testset let d = NormalGamma(μ, λ, α, β)
                ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false)

                (η1, η2, η3, η4) = unpack_parameters(NormalGamma, getnaturalparameters(ef))
                η3half = η3 + 1 / 2
                for x in rand(d, 3)
                    @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                    @test @inferred(basemeasure(ef, x)) ≈ invsqrt2π
                    @test @inferred(sufficientstatistics(ef, x)) === (x[1] * x[2], x[1]^2 * x[2], log(x[2]), x[2])
                    @test @inferred(logpartition(ef)) ≈ loggamma(η3half) - log(-2η2) * (1 / 2) - (η3half) * log(-η4 + η1^2 / (4η2))
                end
            end
        end

        @test !isproper(MeanParametersSpace(), NormalGamma, [1.0, 0.0, -1.0, 2.0])
        @test !isproper(MeanParametersSpace(), NormalGamma, [-1.0, 0.0, NaN, 1.0], [Inf])
    end

    @testset "prod with Distribution" begin
        for strategy in (ClosedProd(), PreserveTypeProd(NormalGamma), PreserveTypeLeftProd(), PreserveTypeRightProd(), GenericProd())
            @test prod(strategy, NormalGamma(1.0, 1.0, 2.0, 3.0), NormalGamma(1.0, 1.0, 5.0, 6.0)) == NormalGamma(1.0, 2.0, 6.5, 9.0)
            @test prod(strategy, NormalGamma(2.0, 1.0, 3.0, 4.0), NormalGamma(2.0, 1.0, 0.4, 2.0)) == NormalGamma(2.0, 2.0, 2.9, 6.0)
        end
    end

    @testset "prod with ExponentialFamilyDistribution" begin
        for μleft in 10randn(2), μright in 10randn(2), σleft in 10rand(2), σright in 10rand(2),
            αleft in 10rand(2), αright in 10rand(2), βleft in 10rand(2), βright in 10rand(2)

            let left = NormalGamma(μleft, σleft, αleft, βleft), right = NormalGamma(μright, σright, αright + 1 / 2, βright)
                @test test_generic_simple_exponentialfamily_product(
                    left,
                    right,
                    strategies = (
                        ClosedProd(),
                        GenericProd(),
                        PreserveTypeProd(ExponentialFamilyDistribution),
                        PreserveTypeProd(ExponentialFamilyDistribution{NormalGamma})
                    )
                )
            end
        end
    end
end

end
