# ExponentialFamily.jl
[![CI][ci-img]][ci-url]

[ci-img]: https://github.com/biaslab/ExponentialFamily.jl/actions/workflows/CI.yml/badge.svg?branch=main
[ci-url]: https://github.com/biaslab/ExponentialFamily.jl/actions

ExponentialFamily.jl is a Julia package that extends the functionality of Distributions.jl by providing a collection of exponential family distributions and customized implementations. It is designed to facilitate working with exponential family distributions and offers specialized functionality tailored to this class of distributions.

## Key Features

- **Distributions**: ExponentialFamily.jl includes a wide range of exponential family distributions such as Gaussian, Gamma, Poisson, Fisher, and more. These distributions are built upon the foundation of Distributions.jl, ensuring compatibility and consistency.

- **Analytic Products**: The package provides support for analytic products of distributions belonging to the same exponential family. These products are useful for conjugate computations in inference algorithms, enabling efficient calculations in Bayesian inference and other probabilistic models.

## Examples

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

### Example 2: Laplace distribution
```julia
using ExponentialFamily
import ExponentialFamily: ExponentialFamilyDistribution

dist_left = Laplace(1.0, 3.0)
dist_right = Laplace(1.0, 4.0)
prod(ClosedProd(), dist_left, dist_right)


# Note that the product of Laplace distraibutions with different location parameters is not a Laplace distribution
# yet it is still a member of the exponential family
dist_left = Laplace(1.0, 3.0)
dist_right = Laplace(3.0, 4.0)
prod(ClosedProd(), dist_left, dist_right)
```

## Installation
ExponentialFamily.jl can be installed through the Julia package manager. In the Julia REPL, type `]` to enter the package manager mode and run:
```julia
pkg> add ExponentialFamily
```
