# [What is the Exponential Family?](@id exponential-family-primer)

!!! note
    This page introduces the exponential family from scratch. For the precise
    API and implementation details, continue to the [interface description](@ref interface).
    If the term "probability distribution" is unfamiliar, start with
    [What is a Probability Distribution?](@ref distributions-primer).

## A common form for many distributions

At first glance the [`Bernoulli`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Bernoulli),
[`Normal`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Normal),
[`Gamma`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Gamma),
and [`Poisson`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Poisson)
distributions look completely different. Remarkably, all of them — and most other
distributions you will encounter — can be written in one and the same shape:

```math
p_X(x \mid \eta) = h(x) \cdot \exp\!\left[\, \eta \cdot T(x) - A(\eta) \,\right]
```

A distribution that can be written this way is said to belong to the
**exponential family**. Each symbol plays a specific role:

- **``\eta`` — the natural parameters.** These are the parameters of the
  distribution, but expressed in a special "canonical" coordinate system rather
  than the human-friendly mean parameters (a probability, a mean, a variance).
- **``T(x)`` — the sufficient statistics.** A function (or several functions) of
  the data. The name reflects a deep fact: ``T(x)`` captures *everything* the data
  can tell you about ``\eta``. Once you know ``T(x)``, the raw data carries no extra
  information about the parameters.
- **``A(\eta)`` — the log partition function** (also called the cumulant
  function). It is the bookkeeping term that guarantees the probabilities
  integrate (or sum) to one. Its derivatives also produce useful quantities such
  as the mean of the sufficient statistics and the Fisher information.
- **``h(x)`` — the base measure.** A non-negative function of the data that does
  not depend on the parameters. For many distributions it is simply `1`.

You do not need to memorise this formula to use the package. The important
takeaway is that *one* template describes a huge collection of distributions, so a
single, generic interface can serve all of them.

## Seeing the pieces in code

`ExponentialFamily.jl` lets you take any supported `Distributions.jl`
distribution and view it in this canonical form. The conversion is done with
`convert`:

```@example ef-primer
using ExponentialFamily, Distributions

bernoulli = Bernoulli(0.25)

ef = convert(ExponentialFamilyDistribution, bernoulli)
```

Now we can inspect each ingredient of the exponential family form. The natural
parameters ``\eta`` are stored in a packed (vectorised) form:

```@example ef-primer
getnaturalparameters(ef)
```

For the `Bernoulli` distribution the single natural parameter is the *log-odds*
``\log\frac{p}{1-p}``, which is a different coordinate system than the probability
``p = 0.25`` we started with. The sufficient statistic ``T(x)`` is just ``x``
itself:

```@example ef-primer
getsufficientstatistics(ef)
```

And the log partition function ``A(\eta)`` is available as a callable function of
the natural parameters:

```@example ef-primer
logpartition(ef)
```

The same pattern works for any supported distribution. The full set of accessor
functions — `getbasemeasure`, `getsufficientstatistics`, `getlogpartition`,
`getgradlogpartition`, `getfisherinformation`, `getsupport`, and more — is
documented on the [interface page](@ref interface).

## Why the exponential family matters

Casting distributions into this shared form is not just mathematical elegance —
it enables operations that are at the heart of probabilistic inference.

### Products of distributions are easy

In Bayesian inference, combining a prior with a likelihood (Bayes' rule) requires
*multiplying* two distributions over the same variable. A wonderful property of
the exponential family is that the product of two members of the same family is
*again* a member of that family, and the natural parameters simply **add**. This
makes the operation exact and cheap:

```@example ef-primer
using BayesBase

prior      = Bernoulli(0.5)
likelihood = Bernoulli(0.6)

posterior = prod(PreserveTypeProd(Distribution), prior, likelihood)
```

For distributions whose product cannot be written back in the original form, the
computation can still be carried out exactly in the exponential family
representation. See the [Examples](@ref examples-product) page for a deeper look.

### Sufficient statistics and analytic attributes

Because the structure is shared, quantities that are otherwise tedious to derive
by hand — the base measure, sufficient statistics, the log partition function,
and the [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) —
are all implemented analytically and exposed through one uniform interface. These
are the building blocks of conjugate inference, variational methods, and
natural-gradient optimisation.

### One interface for many distributions

Finally, the exponential family gives the package a single, generic
`ExponentialFamilyDistribution` type that can represent any of its members. Code
written against this interface automatically works for every supported
distribution.

## Where to go next

- [Comparison with Distributions.jl](@ref comparison-distributions) — how this
  package relates to and extends `Distributions.jl`.
- [The `ExponentialFamilyDistribution` Interface](@ref interface) — the complete
  API reference.
- [Examples](@ref examples) — worked examples of products and attribute
  computations.

If you would like a more formal treatment, the
[Wikipedia article on the exponential family](https://en.wikipedia.org/wiki/Exponential_family)
is an excellent reference, including a
[table](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions)
of the canonical form for many common distributions.
