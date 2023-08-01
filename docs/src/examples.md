## Examples

### Example 1: Bernoulli distribution

```julia
using ExponentialFamily, Distributions
import ExponentialFamily: KnownExponentialFamilyDistribution

dist_left = Bernoulli(0.5)
dist_right = Bernoulli(0.6)
@show dist_prod = prod(ClosedProd(), dist_left, dist_right)

ef_left = convert(KnownExponentialFamilyDistribution, dist_left)
ef_right = convert(KnownExponentialFamilyDistribution, dist_right)
ef_prod = prod(ef_left, ef_right)

@show convert(Bernoulli, ef_prod)
```

### Example 2: Laplace distribution
```julia
using ExponentialFamily
import ExponentialFamily: KnownExponentialFamilyDistribution

dist_left = Laplace(1.0, 3.0)
dist_right = Laplace(1.0, 4.0)
prod(ClosedProd(), dist_left, dist_right)


# Note that the product of Laplace distraibutions with different location parameters is not a Laplace distribution
# yet it is still a member of the exponential family
dist_left = Laplace(1.0, 3.0)
dist_right = Laplace(3.0, 4.0)
prod(ClosedProd(), dist_left, dist_right)
```