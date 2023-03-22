export ExponentialFamilyProduct, ClosedProd, ConditionallyClosedProd, ProdGeneric, GenericLogPdfVectorisedProduct

import Distributions
import Distributions: VariateForm, ValueSupport
import Base: prod, show, showerror

struct ClosedProd end
"""
    ClosedProd

`ClosedProd` is one of the strategies for `prod` function. This strategy uses analytical prod methods but does not constraint a prod to be in any specific form.
It throws an `NoClosedProdException` if no analytical rules is available, use `ProdGeneric` prod strategy to fallback to approximation methods.

Note: `ClosedProd` ignores `missing` values and simply returns the non-`missing` argument. Returns `missing` in case if both arguments are `missing`.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdGeneric`](@ref)
"""


"""
    prod(strategy, left, right)

`prod` function is used to find a product of two probability distrubutions (or any other objects) over same variable (e.g. 𝓝(x|μ_1, σ_1) × 𝓝(x|μ_2, σ_2)).
There are multiple strategies for prod function, e.g. `ClosedProd`, `ProdGeneric` or `ProdPreserveType`.

# Examples:
```jldoctest
using ExponentialFamily

product = prod(ClosedProd(), NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(1.0, 1.0))

mean(product), var(product)

# output
(0.0, 0.5)
```

See also: [`prod_analytical_rule`](@ref), [`ClosedProd`](@ref), [`ProdGeneric`](@ref)
"""
prod(::ClosedProd, left, right) = throw(NoClosedProdException(left, right))

prod(::ClosedProd, ::Missing, right)     = right
prod(::ClosedProd, left, ::Missing)      = left
prod(::ClosedProd, ::Missing, ::Missing) = missing

"""
    NoClosedProdException(left, right)

This exception is thrown in the `prod` function in case if an analytical prod between `left` and `right` is not available or not implemented.

See also: [`ClosedProd`](@ref), [`prod`]
"""
struct NoClosedProdException{L, R} <: Exception
    left  :: L
    right :: R
end

function Base.showerror(io::IO, err::NoClosedProdException)
    print(io, "NoClosedProdException: ")
    print(io, "  No analytical rule available to compute a product of $(err.left) and $(err.right).")
    print(
        io,
        "  Possible fix: implement `prod(::ClosedProd, left::$(typeof(err.left)), right::$(typeof(err.right))) = ...`"
    )
end

export ProdPreserveType, ProdPreserveTypeLeft, ProdPreserveTypeRight

"""
    ProdPreserveType{T}
`ProdPreserveType` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in some specific form.
By default it fallbacks to a `ClosedProd` strategy and converts an output to a prespecified type but can be overwritten for some distributions
for better performance.
See also: [`prod`](@ref), [`ClosedProd`](@ref), [`ProdPreserveTypeLeft`](@ref), [`ProdPreserveTypeRight`](@ref)
"""
struct ProdPreserveType{T} end

ProdPreserveType(::Type{T}) where {T} = ProdPreserveType{T}()

prod(::ProdPreserveType{T}, left, right) where {T} = convert(T, prod(ClosedProd(), left, right))

"""
    ProdPreserveTypeLeft
`ProdPreserveTypeLeft` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in the functional form as `left` argument.
By default it fallbacks to a `ProdPreserveType` strategy and converts an output to a prespecified type but can be overwritten for some distributions
for better performance.
See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdPreserveTypeRight`](@ref)
"""
struct ProdPreserveTypeLeft end

prod(::ProdPreserveTypeLeft, left::L, right) where {L} = prod(ProdPreserveType(L), left, right)

"""
    ProdPreserveTypeRight
`ProdPreserveTypeRight` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in the functional form as `right` argument.
By default it fallbacks to a `ProdPreserveType` strategy and converts an output to a prespecified type but can be overwritten for some distributions
for better performance.
See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdPreserveTypeLeft`](@ref)
"""
struct ProdPreserveTypeRight end

prod(::ProdPreserveTypeRight, left, right::R) where {R} = prod(ProdPreserveType(R), left, right)


struct SemiClosedProd end
struct ConditionallyClosedProd end

"""
    prod_analytical_rule(::Type, ::Type)
Returns either `ProdAnalyticalRuleAvailable` or `ProdAnalyticalRuleUnknown` for two given distribution types.
Returns `ProdAnalyticalRuleUnknown` by default.
See also: [`prod`](@ref), [`ProdAnalytical`](@ref), [`ProdGeneric`](@ref)
"""
prod_analytical_rule(::Type, ::Type) = ProdAnalyticalRuleUnknown()

struct ExponentialFamilyProduct{L, R}
    left  :: L
    right :: R
end


Base.show(io::IO, product::ExponentialFamilyProduct) = print(io, "ExponentialFamilyProduct(", getleft(product), ",", getright(product), ")")

getleft(product::ExponentialFamilyProduct)  = product.left
getright(product::ExponentialFamilyProduct) = product.right

function Distributions.support(product::ExponentialFamilyProduct)
    lsupport = Distributions.support(getleft(product))
    rsupport = Distributions.support(getright(product))
    if lsupport != rsupport
        error("Product $product has different support for left and right entries.")
    end
    return lsupport
end

Distributions.pdf(product::ExponentialFamilyProduct, x)    = Distributions.pdf(getleft(product), x) * Distributions.pdf(getright(product), x)
Distributions.logpdf(product::ExponentialFamilyProduct, x) = Distributions.logpdf(getleft(product), x) + Distributions.logpdf(getright(product), x)

variate_form(::P) where {P <: ExponentialFamilyProduct}           = variate_form(P)
variate_form(::Type{ExponentialFamilyProduct{L, R}}) where {L, R} = _check_dist_product_variate_form(variate_form(L), variate_form(R))

_check_dist_product_variate_form(::Type{F}, ::Type{F}) where {F <: VariateForm}                       = F
_check_dist_product_variate_form(::Type{F1}, ::Type{F2}) where {F1 <: VariateForm, F2 <: VariateForm} = error("ExponentialFamilyProduct has different variate forms for left ($F1) and right ($F2) entries.")

value_support(::P) where {P <: ExponentialFamilyProduct}           = value_support(P)
value_support(::Type{ExponentialFamilyProduct{L, R}}) where {L, R} = _check_dist_product_value_support(value_support(L), value_support(R))

_check_dist_product_value_support(::Type{S}, ::Type{S}) where {S <: ValueSupport}                        = S
_check_dist_product_value_support(::Type{S1}, ::Type{S2}) where {S1 <: ValueSupport, S2 <: ValueSupport} = error("ExponentialFamilyProduct has different value supports for left ($S1) and right ($S2) entries.")

"""
    ProdGeneric{C}

`ProdGeneric` is one of the strategies for `prod` function. This strategy does not fail in case of no analytical rule is available, but simply creates a product tree, there all nodes represent the `prod` function and all leaves are valid `Distribution` object.
This object does not define any statistical properties (such as `mean` or `var` etc) and cannot be used during the inference procedure. However this object plays imporant part in the functional form constraints implementation. 
In a few words this object keeps all the information of a product of messages and propagates this information in the functional form constraint.

`ProdGeneric` has a "fallback" method, which it may or may not use under some circumstances. For example if the `fallback` method is `ClosedProd` (which is the default one) - `ProdGeneric` will try to optimize `prod` tree with analytical solutions where possible.

See also: [`prod`](@ref), [`ExponentialFamilyProduct`](@ref), [`ClosedProd`](@ref), [`ProdPreserveType`](@ref), [`prod_analytical_rule`](@ref), [`GenericLogPdfVectorisedProduct`](@ref)
"""
struct ProdGeneric{C}
    prod_constraint::C
end

Base.show(io::IO, prod::ProdGeneric) = print(io, "ProdGeneric(fallback = ", prod.prod_constraint, ")")

get_constraint(prod_generic::ProdGeneric) = prod_generic.prod_constraint

ProdGeneric() = ProdGeneric(ClosedProd())

prod(::ProdGeneric, ::Missing, right)     = right
prod(::ProdGeneric, left, ::Missing)      = left
prod(::ProdGeneric, ::Missing, ::Missing) = missing

prod(generic::ProdGeneric, left::L, right::R) where {L, R} = prod(generic, prod_analytical_rule(L, R), left, right)

prod(generic::ProdGeneric, ::ClosedProd, left, right) = prod(get_constraint(generic), left, right)
prod(generic::ProdGeneric, ::SemiClosedProd, left, right)   = ExponentialFamilyProduct(left, right)

# In case of ProdPointMass we want to propagate a single `ExponentialFamilyProduct` as much as possible and do not create a big tree of product which will reduce performance significantly
# In this methods the general rule is the folowing: If we see that one of the arguments of `ExponentialFamilyProduct` has the same function form 
# as second argument of `prod` function it is better to try to `prod` them together with `NoConstraint` strategy.
prod(generic::ProdGeneric, left::ExponentialFamilyProduct{L, R}, right::T) where {L, R, T} = prod(generic, prod_analytical_rule(L, T), prod_analytical_rule(R, T), left, right)
prod(generic::ProdGeneric, left::T, right::ExponentialFamilyProduct{L, R}) where {L, R, T} = prod(generic, prod_analytical_rule(T, L), prod_analytical_rule(T, R), left, right)

prod(generic::ProdGeneric, ::SemiClosedProd, ::SemiClosedProd, left::ExponentialFamilyProduct, right)   = ExponentialFamilyProduct(left, right)
prod(generic::ProdGeneric, ::ClosedProd, ::SemiClosedProd, left::ExponentialFamilyProduct, right) = ExponentialFamilyProduct(prod(get_constraint(generic), getleft(left), right), getright(left))
prod(generic::ProdGeneric, ::SemiClosedProd, ::ClosedProd, left::ExponentialFamilyProduct, right) = ExponentialFamilyProduct(getleft(left), prod(get_constraint(generic), getright(left), right))

prod(generic::ProdGeneric, ::SemiClosedProd, ::SemiClosedProd, left, right::ExponentialFamilyProduct)   = ExponentialFamilyProduct(left, right)
prod(generic::ProdGeneric, ::ClosedProd, ::SemiClosedProd, left, right::ExponentialFamilyProduct) = ExponentialFamilyProduct(prod(get_constraint(generic), left, getleft(right)), getright(right))
prod(generic::ProdGeneric, ::SemiClosedProd, ::ClosedProd, left, right::ExponentialFamilyProduct) = ExponentialFamilyProduct(getleft(right), prod(get_constraint(generic), left, getright(right)))

function prod(generic::ProdGeneric, left::ExponentialFamilyProduct{L1, R1}, right::ExponentialFamilyProduct{L2, R2}) where {L1, R1, L2, R2}
    return prod(generic, prod_analytical_rule(L1, L2), prod_analytical_rule(L1, R2), prod_analytical_rule(R1, L2), prod_analytical_rule(R1, R2), left, right)
end

prod(::ProdGeneric, _, _, _, _, left::ExponentialFamilyProduct, right::ExponentialFamilyProduct) = ExponentialFamilyProduct(left, right)

function prod(generic::ProdGeneric, ::ClosedProd, _, _, ::ClosedProd, left::ExponentialFamilyProduct, right::ExponentialFamilyProduct)
    return prod(generic, prod(get_constraint(generic), getleft(left), getleft(right)), prod(get_constraint(generic), getright(left), getright(right)))
end

function prod(left::KnownExponentialFamilyDistribution{T1}, right::KnownExponentialFamilyDistribution{T2}) where {T1,T2}
    return prod(prod_analytical_rule(T1,T2),left,right)
end
prod(::ClosedProd, left::KnownExponentialFamilyDistribution{T}, right::KnownExponentialFamilyDistribution{T}) where {T} =
    KnownExponentialFamilyDistribution(T, getnaturalparameters(left) + getnaturalparameters(right), getconditioner(left))


function prod(::ClosedProd, left::Distribution{T}, right::Distribution{T}) where {T}
    efleft = convert(KnownExponentialFamilyDistribution, left)
    efright = convert(KnownExponentialFamilyDistribution, right)

    return convert(Distribution, prod(efleft,efright))
end


function prod(::ConditionallyClosedProd, left::KnownExponentialFamilyDistribution{T}, right::KnownExponentialFamilyDistribution{T}) where {T}
    leftconditioner = getconditioner(left)
    rightconditioner = getconditioner(right)

    @assert leftconditioner === rightconditioner "Conditioners of $(left) and $(right) must be the same in order for a product to be defined between them."

    return KnownExponentialFamilyDistribution(T, getnaturalparameters(left)+getnaturalparameters(right), getconditioner(left))
end

function prod(::ConditionallyClosedProd,left::Distribution{T},right::Distribution{T}) where {T} 
    efleft = convert(KnownExponentialFamilyDistribution, left)
    efright = convert(KnownExponentialFamilyDistribution, right)

    return convert(Distribution, prod(efleft,efright))
end


