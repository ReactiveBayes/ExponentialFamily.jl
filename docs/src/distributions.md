# [What is a Probability Distribution?](@id distributions-primer)

!!! note
    This page is written for newcomers to probability theory. If you are already
    comfortable with random variables, densities, and parameters, feel free to
    jump straight to [What is the Exponential Family?](@ref exponential-family-primer)
    or to the [interface description](@ref interface).

## Random variables and uncertainty

Many quantities in the real world are *uncertain*: the outcome of a coin flip,
the height of the next person to walk through a door, the number of emails you
will receive tomorrow. A **random variable** is simply a name for such an
uncertain quantity. We usually write it with a capital letter, e.g. ``X``.

A **probability distribution** is the mathematical object that describes *how
likely* each possible value of a random variable is. It answers questions like
"how probable is it that ``X`` equals 3?" or "how probable is it that ``X`` lands
between 1.0 and 2.0?".

Distributions come in two broad flavours:

- **Discrete** distributions describe variables that take values from a countable
  set, such as `0`/`1` (a coin flip) or the non-negative integers (a count).
- **Continuous** distributions describe variables that take values on a continuum,
  such as any real number (a temperature measurement).

## Probability mass and probability density

For a **discrete** random variable, the distribution is summarised by a
*probability mass function* (PMF), written ``p(x)``. It gives the probability that
the variable equals a specific value, and all the probabilities sum to one.

The simplest example is the [`Bernoulli`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Bernoulli)
distribution, which models a single yes/no (or `1`/`0`) event such as a biased
coin flip. It has a single parameter ``p``, the probability of observing a `1`:

```@example dist-primer
using ExponentialFamily, Distributions

coin = Bernoulli(0.25) # a coin that lands "1" 25% of the time

(pdf(coin, 1), pdf(coin, 0)) # probabilities of the two possible outcomes
```

The two numbers add up to one, because `1` and `0` are the only possible
outcomes.

For a **continuous** random variable we cannot talk about the probability of an
*exact* value (that probability is zero), so instead we use a *probability density
function* (PDF), also written ``p(x)``. The density itself is not a probability;
rather, the *area* under the density over an interval gives the probability of
landing in that interval.

The most famous continuous distribution is the
[`Normal`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Normal)
(or Gaussian) distribution, the familiar bell curve:

```@example dist-primer
bell = Normal(0.0, 1.0) # mean 0, standard deviation 1

pdf(bell, 0.0) # the density is highest at the mean
```

## Parameters

Notice that both `Bernoulli(0.25)` and `Normal(0.0, 1.0)` were created by giving
a few numbers. These numbers are the **parameters** of the distribution: they
pin down one specific distribution from a whole family of related shapes. The
`Bernoulli` family is indexed by a single probability ``p``; the `Normal` family
is indexed by a mean ``\mu`` and a variance (or standard deviation) ``\sigma``.

These "human-friendly" parameters — a probability, a mean, a variance — are what
we call the **mean parameters** of a distribution. You can always read them back
from a distribution with `params`, `mean`, and `var`:

```@example dist-primer
(params(bell), mean(bell), var(bell))
```

## Why a package specifically for the *exponential family*?

The [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package
already provides a large collection of distributions together with sampling,
fitting, and density evaluation. So why does `ExponentialFamily.jl` exist?

It turns out that most of the distributions you meet in practice — `Bernoulli`,
`Normal`, `Gamma`, `Poisson`, and many more — share a common mathematical
structure. They all belong to a single, unifying class called the
**exponential family**. Writing a distribution in this common form unlocks
operations that are central to Bayesian inference, most notably *analytic
products* of distributions and direct access to quantities such as sufficient
statistics, the log partition function, and Fisher information.

`ExponentialFamily.jl` builds directly on top of `Distributions.jl` and adds
exactly this machinery. The next page explains what the exponential family is and
why this shared structure is so useful.

**Next:** [What is the Exponential Family?](@ref exponential-family-primer)
