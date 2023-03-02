export vague
export mean, median, mode, shape, scale, rate, var, std, cov, invcov, entropy, pdf, logpdf, logdetcov
export mean_cov, mean_var, mean_std, mean_invcov, mean_precision, weightedmean_cov, weightedmean_var, weightedmean_std, weightedmean_invcov, weightedmean_precision
export weightedmean, probvec, isproper
export variate_form, value_support, promote_variate_type, convert_eltype
export naturalparams, as_naturalparams, lognormalizer, NaturalParameters

using Distributions, Random

import Distributions: mean, median, mode, shape, scale, rate, var, std, cov, invcov, entropy, pdf, logpdf, logdetcov
import Distributions: VariateForm, ValueSupport, Distribution, Univariate, Multivariate, Matrixvariate

import Base: prod,convert

"""
    vague(distribution_type, [ dims... ])

`vague` function returns uninformative probability distribution of a given type and can be used to set an uninformative priors in a model.
"""
function vague end

mean_cov(something)       = (mean(something), cov(something))
mean_var(something)       = (mean(something), var(something))
mean_std(something)       = (mean(something), std(something))
mean_invcov(something)    = (mean(something), invcov(something))
mean_precision(something) = mean_invcov(something)

weightedmean_cov(something)       = (weightedmean(something), cov(something))
weightedmean_var(something)       = (weightedmean(something), var(something))
weightedmean_std(something)       = (weightedmean(something), std(something))
weightedmean_invcov(something)    = (weightedmean(something), invcov(something))
weightedmean_precision(something) = weightedmean_invcov(something)

isproper(something)     = error("`isproper` is not defined for $(something)")
probvec(something)      = error("Probability vector function probvec() is not defined for $(something)")
weightedmean(something) = error("Weighted mean is not defined for $(something)")

"""
    variate_form(distribution_or_type)

Returns the `VariateForm` sub-type (defined in `Distributions.jl`):

- `Univariate`, a scalar number
- `Multivariate`, a numeric vector
- `Matrixvariate`, a numeric matrix

Note: supports real-valued containers, for which it defines:

- `variate_form(real) = Univariate`
- `variate_form(vector_of_reals) = Multivariate`
- `variate_form(matrix_of_reals) = Matrixvariate`

See also: [`ReactiveMP.value_support`](@ref)
"""
variate_form(::Distribution{F, S}) where {F <: VariateForm, S <: ValueSupport} = F
variate_form(::Type{<:Distribution{F, S}}) where {F <: VariateForm, S <: ValueSupport} = F

variate_form(::Type{T}) where {T <: Real} = Univariate
variate_form(::T) where {T <: Real}       = Univariate

variate_form(::Type{V}) where {T <: Real, V <: AbstractVector{T}} = Multivariate
variate_form(::V) where {T <: Real, V <: AbstractVector{T}}       = Multivariate

variate_form(::Type{M}) where {T <: Real, M <: AbstractMatrix{T}} = Matrixvariate
variate_form(::M) where {T <: Real, M <: AbstractMatrix{T}}       = Matrixvariate

# Note that the recent version of `Distributions.jl` has the exact same methods (`variate_form`) with the exact same names, however, old versions do not.
# We keep that for backward-compatibility with old `Distributions.jl` versions,
# but probably we should revise this at some point and remove our implementations (except for the `real` constrained versions)

"""
    value_support(distribution_or_type)

Returns the `ValueSupport` sub-type (defined in `Distributions.jl`):

- `Discrete`, samples take discrete values
- `Continuous`, samples take continuous real values

See also: [`ReactiveMP.variate_form`](@ref)
"""
value_support(::Distribution{F, S}) where {F <: VariateForm, S <: ValueSupport} = S
value_support(::Type{<:Distribution{F, S}}) where {F <: VariateForm, S <: ValueSupport} = S

# Note that the recent version of `Distributions.jl` has the exact same methods (`value_support`) with the exact same names, however, old versions do not
# We keep that for backward-compatibility with old `Distributions.jl` versions, but probably we should revise this at some point and remove our implementations

"""
    promote_variate_type(::Type{ <: VariateForm }, distribution_type)

Promotes (if possible) a `distribution_type` to be of the specified variate form.
"""
function promote_variate_type end

promote_variate_type(::D, T) where {D <: Distribution}       = promote_variate_type(variate_form(D), T)
promote_variate_type(::Type{D}, T) where {D <: Distribution} = promote_variate_type(variate_form(D), T)

"""
    convert_eltype(::Type{D}, ::Type{E}, distribution)

Converts (if possible) the `distribution` to be of type `D{E}`.
"""
convert_eltype(::Type{D}, ::Type{E}, distribution::Distribution) where {D <: Distribution, E} = convert(D{E}, distribution)

"""
    convert_eltype(::Type{E}, container)

Converts (if possible) the elements of the `container` to be of type `E`.
"""
convert_eltype(::Type{E}, container::AbstractArray) where {E} = convert(AbstractArray{E}, container)
convert_eltype(::Type{E}, number::Number) where {E} = convert(E, number)

"""
    sampletype(distribution)

Returns a type of the distribution. By default fallbacks to the `eltype`.

See also: [`ReactiveMP.samplefloattype`](@ref), [`ReactiveMP.promote_sampletype`](@ref), [`ReactiveMP.promotesamplefloatype`](@ref)
"""
sampletype(distribution) = eltype(distribution)

sampletype(distribution::Distribution) = sampletype(variate_form(distribution), distribution)
sampletype(::Type{Univariate}, distribution) = eltype(distribution)
sampletype(::Type{Multivariate}, distribution) = Vector{eltype(distribution)}
sampletype(::Type{Matrixvariate}, distribution) = Matrix{eltype(distribution)}

"""
    samplefloattype(distribution)

Returns a type of the distribution or the underlying float type in case if sample is `Multivariate` or `Matrixvariate`. 
By default fallbacks to the `deep_eltype(sampletype(distribution))`.

See also: [`ReactiveMP.sampletype`](@ref), [`ReactiveMP.promote_sampletype`](@ref), [`ReactiveMP.promote_samplefloatype`](@ref)
"""
samplefloattype(distribution) = deep_eltype(sampletype(distribution))

"""
    promote_sampletype(distributions...)

Promotes `sampletype` of the `distributions` to a single type. See also `promote_type`.

See also: [`ReactiveMP.sampletype`](@ref), [`ReactiveMP.samplefloattype`](@ref), [`ReactiveMP.promote_samplefloattype`](@ref)
"""
promote_sampletype(distributions...) = promote_type(sampletype.(distributions)...)

"""
    promote_samplefloattype(distributions...)

Promotes `samplefloattype` of the `distributions` to a single type. See also `promote_type`.

See also: [`ReactiveMP.sampletype`](@ref), [`ReactiveMP.samplefloattype`](@ref), [`ReactiveMP.promote_sampletype`](@ref)
"""
promote_samplefloattype(distributions...) = promote_type(samplefloattype.(distributions)...)

"""
    logpdf_sample_friendly(distribution) 
    
`logpdf_sample_friendly` function takes as an input a `distribution` and returns corresponding optimized two versions 
for taking `logpdf()` and sampling with `rand!` respectively. By default returns the same distribution, but some distributions 
may override default behaviour for better efficiency.

# Example

```jldoctest
julia> d = vague(MvNormalMeanPrecision, 2)
MvNormalMeanPrecision(
μ: [0.0, 0.0]
Λ: [1.0e-12 0.0; 0.0 1.0e-12]
)


julia> ReactiveMP.logpdf_sample_friendly(d)
(FullNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0e12 -0.0; -0.0 1.0e12]
)
, FullNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0e12 -0.0; -0.0 1.0e12]
)
)
```
"""
logpdf_sample_friendly(something) = (something, something)

"""Abstract type for structures that represent natural parameters of the exponential distributions family"""
abstract type NaturalParameters end

Base.convert(::Type{T}, params::NaturalParameters) where {T <: Distribution} = convert(T, convert(Distribution, params))

"""
    naturalparams(distribution)

Returns the natural parameters for the `distribution`. The `distribution` must be a member of the exponential family of distributions.
"""
function naturalparams end

"""
    as_naturalparams(::Type{T}, args...)

Converts `args` (and promotes if necessary) to the natural parameters ot type `T`. Does not always returns an instance of type `T` but the closes one after type promotion.
"""
function as_naturalparams end

function lognormalizer end

"""
    FactorizedJoint

`FactorizedJoint` represents a joint distribution of independent random variables. Use `getindex()` function or square-brackets indexing to access
the marginal distribution for individual variables.
"""
struct FactorizedJoint{T}
    multipliers::T
end

getmultipliers(joint::FactorizedJoint) = joint.multipliers

Base.getindex(joint::FactorizedJoint, i::Int) = getindex(getmultipliers(joint), i)

Base.length(joint::FactorizedJoint) = length(joint.multipliers)

function Base.isapprox(x::FactorizedJoint, y::FactorizedJoint; kwargs...)
    length(x) === length(y) && all(pair -> isapprox(pair[1], pair[2]; kwargs...), zip(getmultipliers(x), getmultipliers(y)))
end

Distributions.entropy(joint::FactorizedJoint) = mapreduce(entropy, +, getmultipliers(joint))


export ProdAnalytical

import Base: prod, showerror

"""
    ProdAnalytical

`ProdAnalytical` is one of the strategies for `prod` function. This strategy uses analytical prod methods but does not constraint a prod to be in any specific form.
It throws an `NoAnalyticalProdException` if no analytical rules is available, use `ProdGeneric` prod strategy to fallback to approximation methods.

Note: `ProdAnalytical` ignores `missing` values and simply returns the non-`missing` argument. Returns `missing` in case if both arguments are `missing`.

See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdGeneric`](@ref)
"""
struct ProdAnalytical end

"""
    prod(strategy, left, right)

`prod` function is used to find a product of two probability distrubutions (or any other objects) over same variable (e.g. 𝓝(x|μ_1, σ_1) × 𝓝(x|μ_2, σ_2)).
There are multiple strategies for prod function, e.g. `ProdAnalytical`, `ProdGeneric` or `ProdPreserveType`.

# Examples:
```jldoctest
using ReactiveMP

product = prod(ProdAnalytical(), NormalMeanVariance(-1.0, 1.0), NormalMeanVariance(1.0, 1.0))

mean(product), var(product)

# output
(0.0, 0.5)
```

See also: [`prod_analytical_rule`](@ref), [`ProdAnalytical`](@ref), [`ProdGeneric`](@ref)
"""
prod(::ProdAnalytical, left, right) = throw(NoAnalyticalProdException(left, right))

prod(::ProdAnalytical, ::Missing, right)     = right
prod(::ProdAnalytical, left, ::Missing)      = left
prod(::ProdAnalytical, ::Missing, ::Missing) = missing

"""
    NoAnalyticalProdException(left, right)

This exception is thrown in the `prod` function in case if an analytical prod between `left` and `right` is not available or not implemented.

See also: [`ProdAnalytical`](@ref), [`prod`]
"""
struct NoAnalyticalProdException{L, R} <: Exception
    left  :: L
    right :: R
end

function Base.showerror(io::IO, err::NoAnalyticalProdException)
    print(io, "NoAnalyticalProdException: ")
    print(io, "  No analytical rule available to compute a product of $(err.left) and $(err.right).")
    print(io, "  Possible fix: implement `prod(::ProdAnalytical, left::$(typeof(err.left)), right::$(typeof(err.right))) = ...`")
end


export ProdPreserveType, ProdPreserveTypeLeft, ProdPreserveTypeRight


"""
    ProdPreserveType{T}
`ProdPreserveType` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in some specific form.
By default it fallbacks to a `ProdAnalytical` strategy and converts an output to a prespecified type but can be overwritten for some distributions
for better performance.
See also: [`prod`](@ref), [`ProdAnalytical`](@ref), [`ProdPreserveTypeLeft`](@ref), [`ProdPreserveTypeRight`](@ref)
"""
struct ProdPreserveType{T} end

ProdPreserveType(::Type{T}) where {T} = ProdPreserveType{T}()

prod(::ProdPreserveType{T}, left, right) where {T} = convert(T, prod(ProdAnalytical(), left, right))

"""
    ProdPreserveTypeLeft
`ProdPreserveTypeLeft` is one of the strategies for `prod` function. This strategy constraint an output of a prod to be in the functional form as `left` argument.
By default it fallbacks to a `ProdPreserveType` strategy and converts an output to a prespecified type but can be overwritten for some distributions
for better performance.
See also: [`prod`](@ref), [`ProdPreserveType`](@ref), [`ProdPreserveTypeRight`](@ref)
"""
struct ProdPreserveTypeLeft  end

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