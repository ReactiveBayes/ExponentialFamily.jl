# [The `ExponentialFamilyDistribution` Interface](@id interface)

This page describes the philosophy and design concepts behind the `ExponentialFamilyDistribution` interface.
In a nutshell, the primary purpose of the `ExponentialFamily` package is to provide a generic interface for an `ExponentialFamilyDistribution`.
It is beneficial to become familiar with the [Wikipedia article](https://en.wikipedia.org/wiki/Exponential_family) on the exponential family before delving into the implementation details of this package.

## Notation

In the context of the package, exponential family distributions are represented in the form:

```math
f_X(x\mid\eta) = h(x) \cdot \exp\left[ \eta \cdot T(x) - A(\eta) \right]
```

Here:
- `h(x)` is the base measure.
- `T(x)` represents sufficient statistics.
- `A(\theta)` stands for the log partition.
- `\eta` denotes the natural parameters.

In the following discussion we also use the following convention

- `η` corresponds to the distribution's natural parameters in the natural parameter space.
- `θ` corresponds to the distribution's mean parameters in the mean parameter space.

## `ExponentialFamilyDistribution` structure

```@docs 
ExponentialFamilyDistribution
ExponentialFamilyDistributionAttributes
logpdf(ef::ExponentialFamilyDistribution, x)
pdf(ef::ExponentialFamilyDistribution, x)
cdf(ef::ExponentialFamilyDistribution{D}, x) where {D <: Distribution}
getnaturalparameters
getattributes
getconditioner
isproper
getbasemeasure
getsufficientstatistics
getlogpartition
getfisherinformation
getsupport
basemeasure
sufficientstatistics
logpartition
fisherinformation
isbasemeasureconstant
ConstantBaseMeasure
NonConstantBaseMeasure
```

## Interfacing with Distributions Defined in the `Distributions.jl` Package

The [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package is a comprehensive library that defines a wide collection of standard distributions. The main objective of the `Distributions` package is to offer a unified interface for evaluating likelihoods of various distributions, along with convenient sampling routines from these distributions. The `ExponentialFamily` package provides a lightweight interface for a subset of the distributions defined in the `Distributions` package.

### Conversion between Mean Parameters Space and Natural Parameters Space

The `Distributions` package introduces the `params` function, which allows the retrieval of parameters for different distributions. For example:

```@example dist-interfacing
using Distributions, ExponentialFamily

distribution = Bernoulli(0.25)

tuple_of_θ = params(distribution)
```

These parameters are typically defined in what's known as the __mean parameters space__. However, the `ExponentialFamilyDistribution` expects parameters to be in the __natural parameters space__. To facilitate conversion between these two representations, the `ExponentialFamily` package provides two structures:

```@docs
MeanToNatural
NaturalToMean
ExponentialFamily.getmapping
```

To convert from the mean parameters space to the corresponding natural parameters space, you can use the following code:

```@example dist-interfacing
tuple_of_η = MeanToNatural(Bernoulli)(tuple_of_θ)
```

And to convert back:

```@example dist-interfacing
tuple_of_θ = NaturalToMean(Bernoulli)(tuple_of_η)
```

Alternatuvely, the following API is supported 

```@example dist-interfacing
map(MeanParametersSpace() => NaturalParametersSpace(), Bernoulli, tuple_of_θ)
```

```@example dist-interfacing
map(NaturalParametersSpace() => MeanParametersSpace(), Bernoulli, tuple_of_η)
```

While the `ExponentialFamily` package employs the respective mappings where needed, it's also possible to call these functions manually. For instance, the generic implementation of the `convert` function between `ExponentialFamilyDistribution` and `Distribution` is built in terms of `MeanToNatural` and `NaturalToMean`. Moreover, the `convert` function performs checks to ensure that the provided parameters and conditioner are suitable for a specific distribution type.

```@docs
isproper(::Type{T}, parameters, conditioner = nothing) where { T <: Distribution }
```

### Note on the conditioned distributions

For the conditioned distributions, two additional functions `separate_conditioner` and `join_conditioner` are used to separate the conditioner and actual parameters returned from the `Distributions.params` function.

```@docs 
ExponentialFamily.separate_conditioner
ExponentialFamily.join_conditioner
```

For example, `Laplace` distribution defines the functions in the following way

```julia
# `params` are coming from the `Distribution.params(::Laplace)` and return (location, scale)
# The `location`, however is a fixed parameter in the exponential distribution representation of Laplace
# Hence, we return a tuple of tuple of actual parameter and the conditioner
function separate_conditioner(::Type{Laplace}, params)
    location, scale = params
    return ((scale, ), location)
end

# The `join_conditioner` must join the actual parameters and the conditioner in such a way, that it is compatible 
# with the `Laplace` structure from the `Distributions.jl`. In Laplace, the location parameter goes first.
function join_conditioner(::Type{Laplace}, cparams, conditioner) 
    (scale, ) = cparams
    location = conditioner
    return (location, scale)
end
```

In general, all functions defined for the `ExponentialFamilyDistribution`, such as `getlogpartition` or `getbasemeasure` accept an optional `conditioner` parameter, which is assumed to be `nothing`. 
Conditioned distribution implement the "conditioned" versions of such functions by explicitly requiring the `conditioner` parameter, e.g.

```@example dist-interfacing
getsufficientstatistics(Laplace, 1.0) # explicit `conditioner = 1.0`
```

### Efficient packing of the natural parameters into a vectorized form

The `ExponentialFamilyDistribution` type stores its natural parameters in a vectorized, or packed, format. This is done for the sake of efficiency and to enhance compatibility with autodiff packages like `ForwardDiff`, which anticipate a single parameter vector. As a result, the tuple of natural parameters needs to be converted to its corresponding vectorized form and vice versa. To achieve this, the package provides the `flatten_parameters`, `pack_parameters` and `unpack_parameters` functions.

```@docs
ExponentialFamily.flatten_parameters
ExponentialFamily.pack_parameters
ExponentialFamily.unpack_parameters
```

These functions are not exported by default, but it's important to note that the `ExponentialFamilyDistributions` type doesn't actually store the parameter tuple internally. Instead, the `getnaturalparameters` function returns the corresponding vectorized (packed) form of the natural parameters. In general, only the `ExponentialFamily.unpack_parameters` function must be implemented, as others could be implemented in a generic way.

### Attributes of the exponential family distribution based on `Distribution`

The `ExponentialFamilyDistribution{T} where { T <: Distribution }` type encompasses all fundamental attributes of the exponential family, including `basemeasure`, `logpartition`, `sufficientstatistics`, and `fisherinformation`. Furthermore, it's possible to retrieve the actual functions that compute these attributes. For instance, consider the following example:


```@example dist-interfacing
basemeasure_of_bernoilli = getbasemeasure(Bernoulli)

basemeasure_of_bernoilli(0)
```

```@docs
isproper(::Type{T}, parameters, conditioner) where {T <: Distribution}
getbasemeasure(::Type{T}, ::Nothing) where {T <: Distribution}
getsufficientstatistics(::Type{T}, ::Nothing) where { T <: Distribution }
getlogpartition(::Type{T}, _) where { T <: Distribution }
getfisherinformation(::Type{T}, _) where { T <: Distribution }
```

Certain functions require knowledge about which parameter space is being used. By default, the `NaturalParametersSpace` is assumed.

```@example dist-interfacing
getlogpartition(Bernoulli) === getlogpartition(NaturalParametersSpace(), Bernoulli)
```

```@docs 
NaturalParametersSpace
MeanParametersSpace
```

The `isbasemeasureconstant` function is defined for all supported distributions as well.

```@example dist-interfacing
isbasemeasureconstant(Bernoulli)
```

### Extra defined distributions

The package defines a list of extra distributions for a purpose of more efficiency in different circumstances. The list is available [here](@ref library-list-distributions-extra).