using Test, ExponentialFamily, Random, LinearAlgebra, Distributions, FillArrays

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

ExponentialFamily.default_prod_rule(::Type{ObjectWithClosedProd1}, ::Type{ObjectWithClosedProd1}) = PreserveTypeProd(ObjectWithClosedProd1)
ExponentialFamily.default_prod_rule(::Type{ObjectWithClosedProd2}, ::Type{ObjectWithClosedProd2}) = PreserveTypeProd(ObjectWithClosedProd2)
ExponentialFamily.default_prod_rule(::Type{ObjectWithClosedProd1}, ::Type{ObjectWithClosedProd2}) = PreserveTypeProd(ObjectWithClosedProd1)
ExponentialFamily.default_prod_rule(::Type{ObjectWithClosedProd2}, ::Type{ObjectWithClosedProd1}) = PreserveTypeProd(ObjectWithClosedProd2)

prod(::PreserveTypeProd{ObjectWithClosedProd1}, ::ObjectWithClosedProd1, ::ObjectWithClosedProd1) = ObjectWithClosedProd1()
prod(::PreserveTypeProd{ObjectWithClosedProd2}, ::ObjectWithClosedProd2, ::ObjectWithClosedProd2) = ObjectWithClosedProd2()
prod(::PreserveTypeProd{ObjectWithClosedProd1}, ::ObjectWithClosedProd1, ::ObjectWithClosedProd2) = ObjectWithClosedProd1()
prod(::PreserveTypeProd{ObjectWithClosedProd2}, ::ObjectWithClosedProd2, ::ObjectWithClosedProd1) = ObjectWithClosedProd2()

Base.convert(::Type{ObjectWithClosedProd1}, ::ObjectWithClosedProd2) = ObjectWithClosedProd1()
Base.convert(::Type{ObjectWithClosedProd2}, ::ObjectWithClosedProd1) = ObjectWithClosedProd2()

Base.convert(::Type{Int}, ::ObjectWithClosedProd1) = 1
Base.convert(::Type{Int}, ::ObjectWithClosedProd2) = 2

struct ADistributionObject <: ContinuousUnivariateDistribution end

prod(::PreserveTypeProd{Distribution}, ::ADistributionObject, ::ADistributionObject) = ADistributionObject()
