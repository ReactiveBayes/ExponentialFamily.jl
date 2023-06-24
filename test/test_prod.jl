module ExponentialFamilyProdGenericTest

using Test
using ExponentialFamily
using Random
using LinearAlgebra
using Distributions

import ExponentialFamily: KnownExponentialFamilyDistribution,distributiontype, prod, closed_prod_rule, get_constraint

@testset "ProdGeneric" begin
    × = (x, y) -> prod(ProdGeneric(), x, y)

    @testset "ProdGeneric should use ClosedProd where possible" begin
        @test get_constraint(ProdGeneric()) == ClosedProd()

        efber1 = KnownExponentialFamilyDistribution(Bernoulli, log(0.1))
        efber2 = KnownExponentialFamilyDistribution(Bernoulli, log(0.3))
        @test closed_prod_rule(efber1,efber2) == ClosedProd()
        efberprod = prod(ProdGeneric(), efber1, efber2)
        @test distributiontype(efberprod) === Bernoulli

        efnormal1 = KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [0.2, -2.0])
        efnormal2 = KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [1.0, -0.1])
        @test closed_prod_rule(efnormal1, efnormal2) == ClosedProd()
        efnormalprod = prod(ProdGeneric(), efnormal1, efnormal2)
        @test distributiontype(efnormalprod) === NormalWeightedMeanPrecision
    end

    # @testset "ProdGeneric should simplify a product tree if closed form product available for leafes" begin
    #     struct DummyDistribution11 end
    #     struct DummyDistribution12 end
    #     struct DummyDistribution13 end
    #     struct DummyDistribution14 end

    #     ExponentialFamily.closed_prod_rule(::Type{DummyDistribution12}, ::Type{DummyDistribution13}) = ClosedProd()
    #     ExponentialFamily.closed_prod_rule(::Type{DummyDistribution13}, ::Type{DummyDistribution12}) = ClosedProd()
    #     ExponentialFamily.prod(::ClosedProd, ::DummyDistribution12, ::DummyDistribution13)           = DummyDistribution14()
    #     ExponentialFamily.prod(::ClosedProd, ::DummyDistribution13, ::DummyDistribution12)           = DummyDistribution14()

    #     d1 = DummyDistribution11()
    #     d2 = DummyDistribution12()
    #     d3 = DummyDistribution13()
    #     d4 = DummyDistribution14()

    #     @test (d1 × d2) × d3 === d1 × d4
    #     @test (d2 × d1) × d3 === d4 × d1

    #     @test d3 × (d2 × d1) === d4 × d1
    #     @test d3 × (d1 × d2) === d1 × d4

    #     @test (d2 × d2) × (d3 × d3) === (d4 × d4)
    #     @test (d3 × d3) × (d2 × d2) === (d4 × d4)
    # end

    # @testset "ProdGeneric should create a product tree if closed form product is not available" begin
    #     struct DummyDistribution21 end
    #     struct DummyDistribution22 end
    #     struct DummyDistribution23 end

    #     d1 = DummyDistribution21()
    #     d2 = DummyDistribution22()
    #     d3 = DummyDistribution23()

    #     @test d1 × d2 === ExponentialFamilyProduct(DummyDistribution21(), DummyDistribution22())
    #     @test (d1 × d2) × d3 === ExponentialFamilyProduct(ExponentialFamilyProduct(DummyDistribution21(), DummyDistribution22()), DummyDistribution23())
    # end

    # @testset "ProdGeneric should create a linearised product tree if closed form product is not available, but objects are of the same type" begin
    #     struct DummyDistribution31 end
    #     struct DummyDistribution32 end

    #     d1 = DummyDistribution31()
    #     d2 = DummyDistribution32()

    #     @test d1 × d2 === ExponentialFamilyProduct(DummyDistribution31(), DummyDistribution32())
    #     @test d1 × d2 × d2 × d2 isa ExponentialFamilyProduct{DummyDistribution31, ExponentialFamilyProductLogPdf{DummyDistribution32}}
    #     @test (d1 × d2 × d2 × d2) × d1 × d1 isa ExponentialFamilyProduct{ExponentialFamilyProductLogPdf{DummyDistribution31}, ExponentialFamilyProductLogPdf{DummyDistribution32}}
    # end
end

end