# [Examples](@id examples)

## [Product of two probability distributions over the same variable](@id examples-product)

In this example, we'll show you how to use the ExponentialFamily package to calculate the product of probability distributions that both relate to the same variable "X". 
This operation results in another probability distribution, also centered around the variable "X". 
It's essential to note that this is distinct from multiplying probability distributions for two different variables, such as "X" and "Y", 
which yields a joint probability distribution. Calculating the product of two probability distributions over the same variable is a crucial step in applying Bayes' rule.

```math 
p(X\vert D) \propto \underbrace{p(X)p(D|X)}_{\mathrm{product~of~two~distributions}}
```

To perform this operation, the ExponentialFamily library employs the `prod` function. This function takes a product strategy as its first argument. For instance:

```@example prod-example
using ExponentialFamily, Distributions, BayesBase

prior = Bernoulli(0.5)
likelihood = Bernoulli(0.6)

posterior = prod(PreserveTypeProd(Distribution), prior, likelihood)
```

You can achieve the same result in the general exponential family form, which operates within the natural parameter space:

```@example prod-example

ef_prior = convert(ExponentialFamilyDistribution, prior)
ef_likelihood = convert(ExponentialFamilyDistribution, likelihood)

ef_posterior = prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_prior, ef_likelihood)
```

Or even more concisely:

```@example prod-example
prod(PreserveTypeProd(ExponentialFamilyDistribution), prior, likelihood)
```

In this example, multiplying two `Bernoulli` distributions will always result in another `Bernoulli` distribution. However, this is not the case for all distributions. For instance, the product of two `Laplace` distributions may not yield another `Laplace` distribution, and representing the result in the same form might not be possible. In such cases, it's advisable to calculate the result within the exponential family domain. This is because the product of two exponential family distributions can always be represented as another exponential family distribution, as shown here:

```@example prod-example

prior = Laplace(2.0, 3.0)
likelihood = Laplace(1.0, 4.0)

prod(PreserveTypeProd(ExponentialFamilyDistribution), prior, likelihood)
```

Note that the result does not correspond to the `Laplace` distribution and returns a generic univariate `ExponentialFamilyDistribution`.
This approach ensures consistency and compatibility, especially when dealing with a wide range of probability distributions.
Refer to the [`BayesBase`](https://github.com/ReactiveBayes/BayesBase.jl) for the documentation about available product strategies.

## Computing various useful attributes of an exponential family member

The package implements attributes of many well known exponential family members, which are defined in [this table](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions).
The attributes include [`getbasemeasure`](@ref), [`getsufficientstatistics`](@ref), [`getlogpartition`](@ref), [`getfisherinformation`](@ref), and others. 
In general, the interface for these functions assumes a family member "tag," such as `Normal` or `Bernoulli`. Here are some examples of how to use these attributes:

```@example attributes-example
using ExponentialFamily, Distributions

# Returns a function
basemeasure_of_normal = getbasemeasure(Normal)

basemeasure_of_normal(0.0)
```

```@example attributes-example

# Returns an iterable of functions
sufficientstatistics_of_gamma = getsufficientstatistics(Gamma)

map(f -> f(1.0), sufficientstatistics_of_gamma)
```

Some distributions, like the `Laplace` distribution, qualify as exponential family members only under certain conditions or when specific information is known in advance. In such cases, the ExponentialFamily package introduces the concept of a __conditioner__. For instance, the `Laplace` distribution becomes a member of the exponential family only when the `location` parameter is known and fixed. Consequently, we __condition__ on the `location` parameter:

```@example attributes-example
laplace = Laplace(2.0, 1.0)

canonical = convert(ExponentialFamilyDistribution, laplace)

getconditioner(canonical)
```

```@example attributes-example
# For conditioned distributions, the `conditioner` must be present as a second argument
basemeasure_of_laplace = getbasemeasure(Laplace, 2.0)

basemeasure_of_laplace(1.0)
```

The [`getlogpartition`](@ref) and [`getfisherinformation`](@ref) functions optionally accept a `space` parameter as the first argument. This `space` parameter specifies the parameterization `space`, such as [`DefaultParametersSpace`](@ref) or [`NaturalParametersSpace`]. The result obtained from these functions (in general) depends on the chosen parameter space:

```@example attributes-example
logpartition_of_gamma_in_mean_space = getlogpartition(DefaultParametersSpace(), Gamma)

gamma_parameters_in_mean_space = [ 1.0, 2.0 ]

logpartition_of_gamma_in_mean_space(gamma_parameters_in_mean_space)
```

```@example attributes-example
logparition_of_gamma_in_natural_space = getlogpartition(NaturalParametersSpace(), Gamma)

gamma_parameters_in_natural_space = map(
    DefaultParametersSpace() => NaturalParametersSpace(), 
    Gamma,
    gamma_parameters_in_mean_space
)

logparition_of_gamma_in_natural_space(gamma_parameters_in_natural_space)
```

The same principle applies to the Fisher information matrix:

```@example attributes-example
fisherinformation_of_gamma_in_mean_space = getfisherinformation(DefaultParametersSpace(), Gamma)

fisherinformation_of_gamma_in_mean_space(gamma_parameters_in_mean_space)
```

```@example attributes-example
fisherinformation_of_gamma_in_natural_space = getfisherinformation(NaturalParametersSpace(), Gamma)

fisherinformation_of_gamma_in_natural_space(gamma_parameters_in_natural_space)
```

## Approximating attributes 

Refer to the [`ExpectationApproximations.jl`](https://github.com/ReactiveBayes/ExpectationApproximations.jl) package for approximating various attributes of the members of the exponential family.
