# ExponentialFamily.jl

ExponentialFamily.jl is a Julia package that brings together exponential family distributions from Distributions.jl and includes customized implementations for some distributions. It adds exponential family-specific functionality to facilitate their use in primarly but not limieted to RxInfer.jl.

## Key Features

- Collects a variety of exponential family distributions from Distributions.jl and custom implementations in one package.
- Offers exponential family-specific functionality to enhance the user experience.
- Allows for analytic products of distributions belonging to the same exponential family, which can be helpful in conjugate computations for inference algorithms such as CVI.

## Usage

ExponentialFamily.jl aims to provide a convenient way to work with exponential family distributions. The package offers a set of tools to help users incorporate these distributions into their algorithms and applications more easily.

### Multiplying two Binomial distributions

```julia
using ExponentialFamily

# Create two Binomial distributions
binomial1 = Binomial(10, 0.3)
binomial2 = Binomial(10, 0.6)

# Convert them to Exponential Family distributions
ef_binomial1 = convert(KnownExponentialFamilyDistribution, binomial1)
ef_binomial2 = convert(KnownExponentialFamilyDistribution, binomial2)

# Compute the product of the two Exponential Family distributions
ef_binomial_product = prod(ef_binomial1, ef_binomial2)

# Convert the product back to a regular Binomial distribution
binomial_product = convert(Distribution, ef_binomial_product)
