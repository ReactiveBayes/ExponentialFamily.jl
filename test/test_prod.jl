module ExponentialFamilyProdGenericTest

using Test
using ExponentialFamily
using Random
using LinearAlgebra
using Distributions

import ExponentialFamily:
    KnownExponentialFamilyDistribution, distributiontype, prod, closed_prod_rule, get_constraint,
    getleft, getright

@testset "ProdGeneric" begin
    × = (x, y) -> prod(ProdGeneric(), x, y)

    @testset "ProdGeneric should use ClosedProd where possible" begin
        @test get_constraint(ProdGeneric()) == ClosedProd()

        efber1 = KnownExponentialFamilyDistribution(Bernoulli, log(0.1))
        efber2 = KnownExponentialFamilyDistribution(Bernoulli, log(0.3))
        @test closed_prod_rule(efber1, efber2) == ClosedProd()
        efberprod = prod(ProdGeneric(), efber1, efber2)
        @test distributiontype(efberprod) === Bernoulli

        efnormal1 = KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [0.2, -2.0])
        efnormal2 = KnownExponentialFamilyDistribution(NormalWeightedMeanPrecision, [1.0, -0.1])
        @test closed_prod_rule(efnormal1, efnormal2) == ClosedProd()
        efnormalprod = prod(ProdGeneric(), efnormal1, efnormal2)
        @test distributiontype(efnormalprod) === NormalWeightedMeanPrecision

        efgamma1 = KnownExponentialFamilyDistribution(Gamma, [2, -2])
        efgamma2 = KnownExponentialFamilyDistribution(Gamma, [3, -3])
        @test closed_prod_rule(efgamma1, efgamma2) == ClosedProd()
        efgammaprod = prod(ProdGeneric(), efgamma1, efgamma2)
        @test distributiontype(efgammaprod) === Gamma
    end

    @testset "ProdGeneric should simplify a product tree if closed form product available for leaves" begin
        dof = 5
        ef1 = KnownExponentialFamilyDistribution(Chisq, dof / 2 - 1)
        ef2 = KnownExponentialFamilyDistribution(Gamma, [3.2, -2.3])
        ef3 = KnownExponentialFamilyDistribution(Gamma, [1.2, -3.1])
        ef4 = prod(ProdGeneric(), ef2, ef3)

        @test (ef1 × ef2) × ef3 == ef1 × ef4
        @test getleft((ef1 × ef2) × ef3) == getleft(ef1 × ef4)
        @test getright((ef1 × ef2) × ef3) == getright(ef1 × ef4)

        @test (ef2 × ef1) × ef3 == ef4 × ef1
        @test getleft((ef2 × ef1) × ef3) == getleft(ef4 × ef1)
        @test getright((ef2 × ef1) × ef3) == getright(ef4 × ef1)

        @test ef3 × (ef2 × ef1) == ef4 × ef1
        @test getleft(ef3 × (ef2 × ef1)) == getleft(ef4 × ef1)
        @test getright(ef3 × (ef2 × ef1)) == getright(ef4 × ef1)

        @test ef3 × (ef1 × ef2) == ef1 × ef4
        @test getleft(ef3 × (ef1 × ef2)) == getleft(ef1 × ef4)
        @test getright(ef3 × (ef1 × ef2)) == getright(ef1 × ef4)

        @test (ef2 × ef2) × (ef3 × ef3) == (ef3 × ef3) × (ef2 × ef2)
    end

    @testset "ProdGeneric should create a product tree if closed form product is not available" begin
        ef1 = KnownExponentialFamilyDistribution(Gamma, [3.1, -2.0])
        ef2 = KnownExponentialFamilyDistribution(Laplace, 0.1, -2)
        ef3 = KnownExponentialFamilyDistribution(Categorical, [0.1, 0.9])

        @test ef1 × ef2 == ProductDistribution(ef1, ef2)
        @test (ef1 × ef2) × ef3 == ProductDistribution(ProductDistribution(ef1, ef2), ef3)
    end

    @testset "ProdGeneric should create a linearised product tree if closed form product is not available, but objects are of the same type" begin
        struct DummyDistribution1 end
        struct DummyDistribution2 end

        d1 = DummyDistribution1()
        d2 = DummyDistribution2()

        @test d1 × d2 === ProductDistribution(DummyDistribution1(), DummyDistribution2())
        @test d1 × d2 × d2 × d2 isa
              ProductDistribution{DummyDistribution1, LinearizedProductDistribution{DummyDistribution2}}
        @test (d1 × d2 × d2 × d2) × d1 × d1 isa ProductDistribution{
            LinearizedProductDistribution{DummyDistribution1},
            LinearizedProductDistribution{DummyDistribution2}
        }

        ef1 = KnownExponentialFamilyDistribution(Poisson, 10)
        ef2 = KnownExponentialFamilyDistribution(Weibull, -3, 3)
        ef3 = ef2 × ef2
        ef4 = ef1 × ef1
        ef5 = ef2 × ef2 × ef3

        @test ef1 × ef2 == ProductDistribution(ef1, ef2)
        @test ef5 == ProductDistribution(ef3, ef3)

        @test ef1 × ef2 × ef2 × ef3 == ef1 × ef3 × ef3
    end
end

end
