module ExponentialFamilyProdGenericTest

using Test
using ExponentialFamily
using Random
using LinearAlgebra
using Distributions

import ExponentialFamily:
    ExponentialFamilyDistribution, prod, default_prod_rule, ProductOf, LinearizedProductOf, getleft, getright
import ExponentialFamily:
    UnspecifiedProd, PreserveTypeProd, PreserveTypeLeftProd, PreserveTypeRightProd, ClosedProd, GenericProd

## ===========================================================================
## Tests fixtures

# An object, which does not specify any prod rules
struct SomeUnknownObject end

# Two objects that 
# - implement `ClosedProd` between each other 
# - implement `prod` with `ClosedProd` between each other 
# - can be eaily converted between each other
# - can be converted to an `Int`
struct ObjectWithClosedProd1 end
struct ObjectWithClosedProd2 end

ExponentialFamily.default_prod_rule(::Type{ObjectWithClosedProd1}, ::Type{ObjectWithClosedProd1}) = ClosedProd()
ExponentialFamily.default_prod_rule(::Type{ObjectWithClosedProd2}, ::Type{ObjectWithClosedProd2}) = ClosedProd()
ExponentialFamily.default_prod_rule(::Type{ObjectWithClosedProd1}, ::Type{ObjectWithClosedProd2}) = ClosedProd()
ExponentialFamily.default_prod_rule(::Type{ObjectWithClosedProd2}, ::Type{ObjectWithClosedProd1}) = ClosedProd()

prod(::ClosedProd, ::ObjectWithClosedProd1, ::ObjectWithClosedProd1) = ObjectWithClosedProd1()
prod(::ClosedProd, ::ObjectWithClosedProd2, ::ObjectWithClosedProd2) = ObjectWithClosedProd2()
prod(::ClosedProd, ::ObjectWithClosedProd1, ::ObjectWithClosedProd2) = ObjectWithClosedProd1()
prod(::ClosedProd, ::ObjectWithClosedProd2, ::ObjectWithClosedProd1) = ObjectWithClosedProd2()

Base.convert(::Type{ObjectWithClosedProd1}, ::ObjectWithClosedProd2) = ObjectWithClosedProd1()
Base.convert(::Type{ObjectWithClosedProd2}, ::ObjectWithClosedProd1) = ObjectWithClosedProd2()

Base.convert(::Type{Int}, ::ObjectWithClosedProd1) = 1
Base.convert(::Type{Int}, ::ObjectWithClosedProd2) = 2

## ===========================================================================
## Tests

@testset "UnspecifiedProd" begin
    @testset "`default_prod_rule` should return `UnspecifiedProd` for two unknown objects" begin
        @test default_prod_rule(SomeUnknownObject, SomeUnknownObject) === UnspecifiedProd()
    end

    @testset "`missing` should be ignored with the `UnspecifiedProd`" begin
        @test prod(UnspecifiedProd(), missing, SomeUnknownObject()) === SomeUnknownObject()
        @test prod(UnspecifiedProd(), SomeUnknownObject(), missing) === SomeUnknownObject()
        @test prod(UnspecifiedProd(), missing, missing) === missing
    end
end

@testset "ClosedProd" begin
    @testset "`missing` should be ignored with the `ClosedProd`" begin
        struct SomeObject end
        @test prod(ClosedProd(), missing, SomeUnknownObject()) === SomeUnknownObject()
        @test prod(ClosedProd(), SomeUnknownObject(), missing) === SomeUnknownObject()
        @test prod(ClosedProd(), missing, missing) === missing
    end
end

@testset "PreserveTypeProd" begin
    @testset "`missing` should be ignored with the `PreserveTypeProd`" begin
        # Can convert the result of the prod to the desired type
        @test prod(PreserveTypeProd(SomeUnknownObject), missing, SomeUnknownObject()) isa SomeUnknownObject
        @test prod(PreserveTypeProd(SomeUnknownObject), SomeUnknownObject(), missing) isa SomeUnknownObject
        @test prod(PreserveTypeProd(Missing), missing, missing) isa Missing

        # Cannot convert the result of the prod to the desired type
        @test_throws MethodError prod(PreserveTypeProd(SomeUnknownObject), missing, missing)
    end

    @testset "`PreserveTypeLeftProd` should preserve the type of the left argument" begin
        @test prod(PreserveTypeLeftProd(), ObjectWithClosedProd1(), ObjectWithClosedProd2()) isa ObjectWithClosedProd1
        @test prod(PreserveTypeLeftProd(), ObjectWithClosedProd2(), ObjectWithClosedProd1()) isa ObjectWithClosedProd2
    end

    @testset "`PreserveTypeRightProd` should preserve the type of the left argument" begin
        @test prod(PreserveTypeRightProd(), ObjectWithClosedProd1(), ObjectWithClosedProd2()) isa ObjectWithClosedProd2
        @test prod(PreserveTypeRightProd(), ObjectWithClosedProd2(), ObjectWithClosedProd1()) isa ObjectWithClosedProd1
    end

    @testset "`ProdPreserveType(T)` should preserve the desired type of `T`" begin
        @test prod(PreserveTypeProd(ObjectWithClosedProd1), ObjectWithClosedProd1(), ObjectWithClosedProd1()) isa
              ObjectWithClosedProd1
        @test prod(PreserveTypeProd(ObjectWithClosedProd1), ObjectWithClosedProd1(), ObjectWithClosedProd2()) isa
              ObjectWithClosedProd1
        @test prod(PreserveTypeProd(ObjectWithClosedProd1), ObjectWithClosedProd2(), ObjectWithClosedProd1()) isa
              ObjectWithClosedProd1
        @test prod(PreserveTypeProd(ObjectWithClosedProd1), ObjectWithClosedProd2(), ObjectWithClosedProd2()) isa
              ObjectWithClosedProd1

        @test prod(PreserveTypeProd(ObjectWithClosedProd2), ObjectWithClosedProd1(), ObjectWithClosedProd1()) isa
              ObjectWithClosedProd2
        @test prod(PreserveTypeProd(ObjectWithClosedProd2), ObjectWithClosedProd1(), ObjectWithClosedProd2()) isa
              ObjectWithClosedProd2
        @test prod(PreserveTypeProd(ObjectWithClosedProd2), ObjectWithClosedProd2(), ObjectWithClosedProd1()) isa
              ObjectWithClosedProd2
        @test prod(PreserveTypeProd(ObjectWithClosedProd2), ObjectWithClosedProd2(), ObjectWithClosedProd2()) isa
              ObjectWithClosedProd2

        # The output can be converted to an `Int` (see the fixtures above)
        @test prod(PreserveTypeProd(Int), ObjectWithClosedProd1(), ObjectWithClosedProd1()) isa Int
        @test prod(PreserveTypeProd(Int), ObjectWithClosedProd1(), ObjectWithClosedProd2()) isa Int
        @test prod(PreserveTypeProd(Int), ObjectWithClosedProd2(), ObjectWithClosedProd1()) isa Int
        @test prod(PreserveTypeProd(Int), ObjectWithClosedProd2(), ObjectWithClosedProd2()) isa Int

        # The output can not be converted to a `Float` (see the fixtures above)
        @test_throws MethodError prod(PreserveTypeProd(Float64), ObjectWithClosedProd1(), ObjectWithClosedProd1())
        @test_throws MethodError prod(PreserveTypeProd(Float64), ObjectWithClosedProd1(), ObjectWithClosedProd2())
        @test_throws MethodError prod(PreserveTypeProd(Float64), ObjectWithClosedProd2(), ObjectWithClosedProd1())
        @test_throws MethodError prod(PreserveTypeProd(Float64), ObjectWithClosedProd2(), ObjectWithClosedProd2())
    end
end

@testset "GenericProd" begin
    × = (x, y) -> prod(GenericProd(), x, y)

    @testset "GenericProd should use `default_prod_rule` where possible" begin

        # `SomeUnknownObject` does not implement any prod rule (see the fixtures above)
        @test SomeUnknownObject() × SomeUnknownObject() isa ProductOf{SomeUnknownObject, SomeUnknownObject}
        @test ObjectWithClosedProd1() × SomeUnknownObject() isa ProductOf{ObjectWithClosedProd1, SomeUnknownObject}
        @test SomeUnknownObject() × ObjectWithClosedProd1() isa ProductOf{SomeUnknownObject, ObjectWithClosedProd1}

        @test getleft(ObjectWithClosedProd1() × SomeUnknownObject()) === ObjectWithClosedProd1()
        @test getright(ObjectWithClosedProd1() × SomeUnknownObject()) === SomeUnknownObject()
        @test getleft(SomeUnknownObject() × ObjectWithClosedProd1()) === SomeUnknownObject()
        @test getright(SomeUnknownObject() × ObjectWithClosedProd1()) === ObjectWithClosedProd1()

        # Both `ObjectWithClosedProd1` and `ObjectWithClosedProd2` implement `ClosedProd` as a default (see the fixtures above)
        @test ObjectWithClosedProd1() × ObjectWithClosedProd1() isa ObjectWithClosedProd1
        @test ObjectWithClosedProd2() × ObjectWithClosedProd2() isa ObjectWithClosedProd2
    end

    @testset "ProdGeneric should simplify a product tree if closed form product available for leaves" begin
        d1 = SomeUnknownObject()
        d2 = ObjectWithClosedProd1()
        d3 = ObjectWithClosedProd2()

        @test (d1 × d2) × d2 == d1 × d2
        @test (d1 × d3) × d3 == d1 × d3
        @test (d2 × d3) × d3 == d2
        @test (d3 × d2) × d2 == d3

        @test d1 × (d2 × d2) == d1 × d2
        @test d1 × (d3 × d3) == d1 × d3
        @test d2 × (d3 × d3) == d2
        @test d3 × (d2 × d2) == d3

        @test (d2 × d1) × d2 == (d2 × d1)
        @test (d3 × d1) × d3 == (d3 × d1)
        @test (d2 × d2) × d1 == (d2 × d1)
        @test (d3 × d3) × d1 == (d3 × d1)

        @test d2 × (d1 × d2) == (d1 × d2)
        @test d3 × (d1 × d3) == (d1 × d3)
        @test d2 × (d2 × d1) == (d2 × d1)
        @test d3 × (d3 × d1) == (d3 × d1)
    end

    @testset "ProdGeneric should create a product tree if closed form product is not available" begin
        d1 = SomeUnknownObject()

        @test 1.0 × 1 × d1 isa ProductOf{ProductOf{Float64, Int}, SomeUnknownObject}
        @test 1 × 1.0 × d1 isa ProductOf{ProductOf{Int, Float64}, SomeUnknownObject}
    end

    @testset "ProdGeneric should create a linearised product tree if closed form product is not available, but objects are of the same type" begin
        d1 = SomeUnknownObject()
        d2 = ObjectWithClosedProd1()

        @test d1 × d1 isa ProductOf{SomeUnknownObject, SomeUnknownObject}

        @testset let product = d1 × d1 × d1
            @test product isa LinearizedProductOf{SomeUnknownObject}
            @test length(product) === 3

            # Test that the next prod rule should preserve the type of the linearized product
            @test default_prod_rule(product, d1) isa PreserveTypeProd{LinearizedProductOf{SomeUnknownObject}}
        end

        @testset let product = (d1 × d1 × d1) × d1
            @test product isa LinearizedProductOf{SomeUnknownObject}
            @test length(product) === 4

            # Test that the next prod rule should preserve the type of the linearized product
            @test default_prod_rule(product, d1) isa PreserveTypeProd{LinearizedProductOf{SomeUnknownObject}}
        end

        @test d2 × d1 × d1 × d1 isa ProductOf{ObjectWithClosedProd1, LinearizedProductOf{SomeUnknownObject}}
        @test d2 × d1 × d1 × d1 × d1 isa ProductOf{ObjectWithClosedProd1, LinearizedProductOf{SomeUnknownObject}}

        # d2 × (...) × d2 should fold if closed prod is available
        @test d2 × d1 × d1 × d1 × d1 × d2 == (d2 × d2) × d1 × d1 × d1 × d1
        @test d2 × d1 × d2 × d1 × d2 × d1 × d1 × d2 == (d2 × d2 × d2 × d2) × d1 × d1 × d1 × d1
    end
end

end
