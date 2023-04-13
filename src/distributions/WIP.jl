using ExponentialFamily
import ExponentialFamily: prod, convert, KnownExponentialFamilyDistribution
using Distributions

# Create two Binomial distributions
binomial1 = Binomial(10, 0.3)
binomial2 = Binomial(10, 0.6)
prod(ClosedProd(), binomial1, binomial2)

# Convert them to Exponential Family distributions
ef_binomial1 = convert(KnownExponentialFamilyDistribution, binomial1)
ef_binomial2 = convert(KnownExponentialFamilyDistribution, binomial2)

# Compute the product of the two Exponential Family distributions
binomial_product = prod(ClosedProd(), binomial1, binomial2)


prod(ClosedProd(), Gamma(1, 1), Gamma(1, 1))