# [Examples](@id examples)
In these examples, we demonstrate how to use ExponentialFamily.jl to compute the product of probability distributions.
### Example 1: Bernoulli distribution

```julia
using ExponentialFamily, Distributions
import ExponentialFamily: ExponentialFamilyDistribution

dist_left = Bernoulli(0.5)
dist_right = Bernoulli(0.6)
@show dist_prod = prod(ClosedProd(), dist_left, dist_right)

ef_left = convert(ExponentialFamilyDistribution, dist_left)
ef_right = convert(ExponentialFamilyDistribution, dist_right)
ef_prod = prod(ef_left, ef_right)

@show convert(Bernoulli, ef_prod)
```

We created two Bernoulli distributions and compute their product using the `ClosedProd` trait. We also show how to convert the `Bernoulli` distributions to `ExponentialFamilyDistribution`, compute the product of the exponential family distributions, and convert the product back to a `Bernoulli` distribution.

### Example 2: Laplace distribution
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