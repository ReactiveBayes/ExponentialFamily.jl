# [Examples](@id examples)

## [Product of two probability distributions over the same variable](@id examples-product)

In this example, we'll show you how to use the ExponentialFamily package to calculate the product of probability distributions that both relate to the same variable "X". 
This operation results in another probability distribution, also centered around the variable "X". 
It's essential to note that this is distinct from multiplying probability distributions for two different variables, such as "X" and "Y", 
which yields a joint probability distribution. Calculating the product of two probability distributions over the same variable is a crucial step in applying Bayes' rule.

```math 
p(X\vert D) \propto \underbrace{p(X)p(D|X)}_{\mathrm{product~of~two~distributions}}
```

To perform this operation, the ExponentialFamily library employs the `prod` function. This function takes a product [strategy](@ref api-todo-replace-strategies) as its first argument. For instance:

```@example prod-example
using ExponentialFamily, Distributions

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

## Computing various useful attributes of an exponential family member

Section TODO: write about
- getter functions, like `getlogpartition`, note conditioned distributions
- actual functions, like `logpartition`
- different parameters spaces

## Approximating attributes 

Section TODO: refer to the `ExpectationApproximations.jl` instead.