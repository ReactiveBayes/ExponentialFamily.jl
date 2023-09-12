# [Examples](@id examples)

## Product of two probability distributions over the same variable

In this example, we demonstrate how to use `ExponentialFamily`` to compute the product of probability distributions over the same variable "X", which yields another probability distribution also over the same variable "X". Note that, this operation is different from the product of probability distributions over two different variables "X" and "Y", which yields a joint probability distribution. Computing the product of two probability distributions over the same variable is important for the Bayes rule

```math 
p(X\vert D) \propto \underbrace{p(X)p(D|X)}_{\mathrm{product~of~two~distributions}}
```

In order to compute the product of two distributions, the `ExponentialFamily` library uses the `Base.prod` function, which accepts a product strategy as its first argument, for example

```@example prod-bernoulli
using ExponentialFamily, Distributions

prior = Bernoulli(0.5)
likelihood = Bernoulli(0.6)

posterior = prod(PreserveTypeProd(Distribution), prior, likelihood)
```

We can perform the same operation in the generic exponential family form, which operates in the natural parameter space, for example

```@example prod-bernoulli

ef_prior = convert(ExponentialFamilyDistribution, prior)
ef_likelihood = convert(ExponentialFamilyDistribution, likelihood)

ef_posterior = prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_prior, ef_likelihood)
```

or simply

```@example prod-bernoulli
prod(PreserveTypeProd(ExponentialFamilyDistribution), prior, likelihood)
```

### Laplace distribution

```julia
using ExponentialFamily
import ExponentialFamily: ExponentialFamilyDistribution

dist_left = Laplace(1.0, 3.0)
dist_right = Laplace(1.0, 4.0)
prod(ClosedProd(), dist_left, dist_right)


# Note that the product of Laplace distributions with different location parameters is not a Laplace distribution
# However, it is still a member of the exponential family
dist_left = Laplace(1.0, 3.0)
dist_right = Laplace(3.0, 4.0)
prod(ClosedProd(), dist_left, dist_right)
```

We create two `Laplace` distributions and compute their product using the `ClosedProd` function. We also note that the product of `Laplace` distributions with different location parameters is not a `Laplace` distribution but still a member of the exponential family.


## Other examples

```julia
using ExponentialFamily
import ExponentialFamily: ExponentialFamilyDistribution, fisherinformation

## Multivariate Normal example
dist = MvNormalMeanCovariance([1.0, 1.0], [1.0 0.0; 0.0 1.0])
@show fisherinformation(ef)
@show pdf(dist, [1.0, 1.0])

## LogNormal example
dist = LogNormal(0.0, 1.0)
ef = convert(ExponentialFamilyDistribution, dist)
@show fisherinformation(ef)
@show fisherinformation(dist)
@show pdf(ef, 2)

## Poisson example
ef = ExponentialFamilyDistribution(Poisson, [1.0])
@show fisherinformation(ef)
@show pdf(ef, 2)

```