export ProductDistribution, ClosedProd, ClosedProd, ProdGeneric, LinearizedProductDistribution
export ProdPreserveType, ProdPreserveTypeLeft, ProdPreserveTypeRight

import Distributions
import Base: prod, show, showerror

"""
    ClosedProd

`ClosedProd` is one of the strategies for `prod` function. This strategy uses analytical prod methods but does not constraint a prod to be in any specific form.
It throws a `MethodError` if no analytical rule is available, use `ProdGeneric` prod strategy to fallback to approximation methods.

Note: `ClosedProd` ignores `missing` values and simply returns the non-`missing` argument. Returns `missing` in case if both arguments are `missing`.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdGeneric`](@ref)
"""
struct ClosedProd end

"""
    prod(strategy, left, right)

`prod` function is used to find a product of two probability distributions (or any other objects) over same variable (e.g. 𝓝(x|μ_1, σ_1) × 𝓝(x|μ_2, σ_2)).
There are multiple strategies for prod function, e.g. `ClosedProd`, `GenericProd` or `PreserveTypeProd`.

# Examples:
```jldoctest
using ExponentialFamily

product = prod(ClosedProd(), NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(1.0, 1.0))

mean(product), var(product)

# output
(0.0, 0.5)
```

See also: [`default_prod_rule`](@ref), [`ClosedProd`](@ref), [`GenericProd`](@ref), [`PreserveTypeProd`](@ref)
"""
prod(::ClosedProd, left, right) = throw(MethodError(prod, (ClosedProd(), left, right)))

prod(::ClosedProd, ::Missing, right)     = right
prod(::ClosedProd, left, ::Missing)      = left
prod(::ClosedProd, ::Missing, ::Missing) = missing

"""
    PreserveTypeProd{T}

`PreserveTypeProd` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in some specific form.
By default it uses the strategy from `default_prod_rule` and converts the output to the prespecified type but can be overwritten 
for some distributions for better performance.

See also: [`prod`](@ref), [`ClosedProd`](@ref), [`PreserveTypeLeftProd`](@ref), [`PreserveTypeRightProd`](@ref)
"""
struct PreserveTypeProd{T} end

PreserveTypeProd(::Type{T}) where {T} = PreserveTypeProd{T}()

prod(::PreserveTypeProd{T}, left, right) where {T} = convert(T, prod(default_prod_rule(left, right), left, right))

"""
    PreserveTypeLeftProd

An alias for the `PreserveTypeProd(L)` where `L` is the type of the `left` argument of the `prod` function.

See also: [`prod`](@ref), [`PreserveTypeProd`](@ref), [`PreserveTypeRightProd`](@ref)
"""
struct PreserveTypeLeftProd end

prod(::PreserveTypeLeftProd, left::L, right) where {L} = prod(PreserveTypeProd(L), left, right)

"""
    PreserveTypeRightProd

An alias for the `PreserveTypeProd(R)` where `R` is the type of the `right` argument of the `prod` function.    

See also: [`prod`](@ref), [`PreserveTypeProd`](@ref), [`PreserveTypeLeftProd`](@ref)
"""
struct PreserveTypeRightProd end

prod(::PreserveTypeRightProd, left, right::R) where {R} = prod(PreserveTypeProd(R), left, right)

"""
    UnspecifiedProd

A strategy for the `prod` function, which does not compute the `prod`, but instead fails in run-time and prints a descriptive error message.

See also: [`prod`](@ref), [`ClosedProd`](@ref), [`GenericProd`](@ref)
"""
struct UnspecifiedProd end

prod(::UnspecifiedProd, ::Missing, right)     = right
prod(::UnspecifiedProd, left, ::Missing)      = left
prod(::UnspecifiedProd, ::Missing, ::Missing) = missing

"""
    default_prod_rule(::Type, ::Type)

Returns the most suitable `prod` rule for two given distribution types.
Returns `UnspecifiedProd` by default.

See also: [`prod`](@ref), [`ClosedProd`](@ref), [`GenericProd`](@ref)
"""
default_prod_rule(::Type, ::Type) = UnspecifiedProd()

default_prod_rule(not_a_type, ::Type{R}) where {R} = default_prod_rule(typeof(not_a_type), R)
default_prod_rule(::Type{L}, not_a_type) where {L} = default_prod_rule(L, typeof(not_a_type))
default_prod_rule(not_a_type_left, not_a_type_right) =
    default_prod_rule(typeof(not_a_type_left), typeof(not_a_type_right))

"""
    fuse_supports(left, right)

Fuse supports of two distributions of `left` and `right`.
By default, checks that the supports are identical and throws an error otherwise.
Can implement specific fusions for specific distributions.

See also: [`prod`](@ref), [`ProductOf`](@ref)
"""
function fuse_supports(left, right)
    if !isequal(support(left), support(right))
        error("Cannot form a `ProductOf` $(left) & `$(right)`. Support is incompatible.")
    end
    return support(left)
end

"""
    ProductOf

A generic structure representing a product of two distributions. 
Can be viewed as a tuple of `(left, right)`. 
Does not check nor supports neither variate forms during the creation stage.
Uses the `fuse_support` function to fuse supports of two different distributions.

This object does not define any statistical properties (such as `mean` or `var` etc) and cannot be used as a distribution explicitly.
Instead, it must be further approximated as a member of some other distribution. 

See also: [`prod`](@ref), [`GenericProd`](@ref), [`fuse_support`](@ref)
"""
struct ProductOf{L, R}
    left  :: L
    right :: R
end

getleft(product::ProductOf)  = product.left
getright(product::ProductOf) = product.right

function Base.:(==)(left::ProductOf, right::ProductOf)
    return (getleft(left) == getleft(right)) && (getright(left) == getright(right))
end

function Base.show(io::IO, product::ProductOf)
    print(io, "ProductOf(", getleft(product), ",", getright(product), ")")
end

function support(product::ProductOf)
    return fuse_supports(getleft(product), getright(product))
end

Distributions.pdf(product::ProductOf, x)    = exp(logpdf(product, x))
Distributions.logpdf(product::ProductOf, x) = Distributions.logpdf(getleft(product), x) + Distributions.logpdf(getright(product), x)

variate_form(::P) where {P <: ProductOf} = variate_form(P)
variate_form(::Type{ProductOf{L, R}}) where {L, R} = _check_product_variate_form(variate_form(L), variate_form(R))

_check_product_variate_form(::Type{F}, ::Type{F}) where {F <: VariateForm}                       = F
_check_product_variate_form(::Type{F1}, ::Type{F2}) where {F1 <: VariateForm, F2 <: VariateForm} = error("`ProductOf` has different variate forms for left ($F1) and right ($F2) entries.")

value_support(::P) where {P <: ProductOf}           = value_support(P)
value_support(::Type{ProductOf{L, R}}) where {L, R} = _check_product_value_support(value_support(L), value_support(R))

_check_product_value_support(::Type{S}, ::Type{S}) where {S <: ValueSupport}                        = S
_check_product_value_support(::Type{S1}, ::Type{S2}) where {S1 <: ValueSupport, S2 <: ValueSupport} = error("`ProductOf` has different value supports for left ($S1) and right ($S2) entries.")

"""
    GenericProd

`GenericProd` is one of the strategies for `prod` function. This strategy does always produces a result, 
even if the closed form product is not availble, in which case simply returns the `ProductOf` object. `GenericProd` sometimes 
fallbacks to the `default_prod_rule` which it may or may not use under some circumstances. 
For example if the `default_prod_rule` is `ClosedProd` - `ProdGeneric` will try to optimize the tree with 
analytical closed solutions (if possible).

See also: [`prod`](@ref), [`ProductOf`](@ref), [`ClosedProd`](@ref), [`PreserveTypeProd`](@ref), [`default_prod_rule`](@ref)
"""
struct GenericProd end

Base.show(io::IO, ::GenericProd) = print(io, "ProdGeneric()")

prod(::GenericProd, ::Missing, right)     = right
prod(::GenericProd, left, ::Missing)      = left
prod(::GenericProd, ::Missing, ::Missing) = missing

prod(::GenericProd, left::L, right::R) where {L, R} = prod(GenericProd(), default_prod_rule(L, R), left, right)

prod(::GenericProd, specified_prod, left, right) = prod(specified_prod, left, right)
prod(::GenericProd, ::UnspecifiedProd, left, right) = ProductOf(left, right)

# Try to fuse the tree with analytical solutions (if possible)
# Case (L × R) × T
prod(::GenericProd, left::ProductOf{L, R}, right::T) where {L, R, T} =
    prod(GenericProd(), default_prod_rule(L, T), default_prod_rule(R, T), left, right)

# (L × R) × T cannot be fused, simply return the `ProductOf`
prod(::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left::ProductOf, right) = ProductOf(left, right)

# (L × R) × T can be fused efficiently as (L × T) × R, because L × T has defined the `something` default prod
prod(::GenericProd, something, ::UnspecifiedProd, left::ProductOf, right) =
    ProductOf(prod(something, getleft(left), right), getright(left))

# (L × R) × T can be fused efficiently as L × (R × T), because R × T has defined the `something` default prod
prod(::GenericProd, ::UnspecifiedProd, something, left::ProductOf, right) =
    ProductOf(getleft(left), prod(something, getright(left), right))

# (L × R) × T can be fused efficiently as L × (R × T), because both L × T and R × T has defined the `something` default prod, but we choose R × T
prod(::GenericProd, _, something, left::ProductOf, right) =
    ProductOf(getleft(left), prod(something, getright(left), right))

# Case T × (L × R)
prod(::GenericProd, left::T, right::ProductOf{L, R}) where {L, R, T} =
    prod(GenericProd(), default_prod_rule(T, L), default_prod_rule(T, R), left, right)

# T × (L × R) cannot be fused, simply return the `ProductOf`
prod(::GenericProd, ::UnspecifiedProd, ::UnspecifiedProd, left, right::ProductOf) = ProductOf(left, right)

# T × (L × R) can be fused efficiently as (T × L) × R, because T × L has defined the `something` default prod
prod(::GenericProd, something, ::UnspecifiedProd, left, right::ProductOf) =
    ProductOf(prod(something, left, getleft(right)), getright(right))

# T × (L × R) can be fused efficiently as L × (T × R), because T × R has defined the `something` default prod
prod(::GenericProd, ::UnspecifiedProd, something, left, right::ProductOf) =
    ProductOf(getleft(right), prod(something, left, getright(right)))

# T × (L × R) can be fused efficiently as L × (T × R), because both T × L and T × R has defined the `something` default prod, but we choose T × L
prod(::GenericProd, something, _, left, right::ProductOf) =
    ProductOf(prod(something, left, getleft(right)), getright(right))

# TODO we can extend this logic to `ProductOf × ProductOf`, it would require to define many many methods though

"""
    LinearizedProductOf

An efficient __linearized__ implementation of product of multiple distributions.
This structure prevents `ProductOf` tree from growing too much in case of identical objects. 
This trick significantly reduces Julia compilation times when closed product rules are not available but distributions are of the same type.
Essentially this structure linearizes leaves of the `ProductOf` tree in case if it sees objects of the same type (via dispatch).

See also: [`ProductOf`](@ref), [`GenericProd`]
"""
struct LinearizedProductOf{F}
    vector::Vector{F}
    length::Int # `length` here is needed for extra safety as we implicitly mutate `vector` in `prod`
end

function Base.push!(product::LinearizedProductOf{F}, item::F) where {F}
    vector  = product.vector
    vlength = length(vector)
    return LinearizedProductOf(push!(vector, item), vlength + 1)
end

support(dist::LinearizedProductOf) = support(first(dist.vector))

Base.length(product::LinearizedProductOf) = product.length
Base.eltype(product::LinearizedProductOf) = eltype(first(product.vector))

Base.:(==)(left::LinearizedProductOf, right::LinearizedProductOf) =
    (left.length == right.length) && (left.vector == right.vector)

samplefloattype(product::LinearizedProductOf) = samplefloattype(first(product.vector))

variate_form(::Type{<:LinearizedProductOf{F}}) where {F} = variate_form(F)
variate_form(::LinearizedProductOf{F}) where {F}         = variate_form(F)

value_support(::Type{<:LinearizedProductOf{F}}) where {F} = value_support(F)
value_support(::LinearizedProductOf{F}) where {F}         = value_support(F)

Base.show(io::IO, product::LinearizedProductOf{F}) where {F} =
    print(io, "LinearizedProductOf(", F, ", length = ", product.length, ")")

Distributions.logpdf(dist::LinearizedProductOf, x) =
    mapreduce((d) -> logpdf(d, x), +, view(dist.vector, 1:min(dist.length, length(dist.vector))))

Distributions.pdf(dist::LinearizedProductOf, x) = exp(logpdf(dist, x))

# We assume that it is better (really) to preserve the type of the `LinearizedProductOf`, it is just faster for the compiler
default_prod_rule(::Type{F}, ::Type{LinearizedProductOf{F}}) where {F} = PreserveTypeProd(LinearizedProductOf{F})
default_prod_rule(::Type{LinearizedProductOf{F}}, ::Type{F}) where {F} = PreserveTypeProd(LinearizedProductOf{F})

function prod(::PreserveTypeProd{LinearizedProductOf{F}}, product::LinearizedProductOf{F}, item::F) where {F}
    return push!(product, item)
end

function prod(::PreserveTypeProd{LinearizedProductOf{F}}, item::F, product::LinearizedProductOf{F}) where {F}
    return push!(product, item)
end

function prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::ProductOf{F, F},
    right::F
) where {F}
    return LinearizedProductOf(F[getleft(left), getright(left), right], 3)
end

function prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::ProductOf{L, R},
    right::R
) where {L, R}
    return ProductOf(getleft(left), LinearizedProductOf(R[getright(left), right], 2))
end

function prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::ProductOf{L, R},
    right::L
) where {L, R}
    return ProductOf(LinearizedProductOf(L[getleft(left), right], 2), getright(left))
end

function prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::L,
    right::ProductOf{L, R}
) where {L, R}
    return ProductOf(LinearizedProductOf(L[left, getleft(right)], 2), getright(right))
end

function prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::R,
    right::ProductOf{L, R}
) where {L, R}
    return ProductOf(getleft(right), LinearizedProductOf(R[left, getright(right)], 2))
end

function prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::ProductOf{L, LinearizedProductOf{R}},
    right::R
) where {L, R}
    return ProductOf(getleft(left), push!(getright(left), right))
end

function prod(
    ::GenericProd,
    ::UnspecifiedProd,
    ::UnspecifiedProd,
    left::ProductOf{LinearizedProductOf{L}, R},
    right::L
) where {L, R}
    return ProductOf(push!(getleft(left), right), getright(left))
end
