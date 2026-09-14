# [Comparison with Distributions.jl](@id comparison-distributions)

A common first question is: *how does `ExponentialFamily.jl` relate to
[`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl)?* The short
answer is that **`ExponentialFamily.jl` extends `Distributions.jl` — it does not
replace it.** `Distributions.jl` is a direct dependency, and the two are designed
to be used together.

## What each package is for

[`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) is the
foundational Julia package for probability distributions. It provides a broad
"zoo" of distributions together with a unified interface for the operations you
most often need:

- **sampling** random values (`rand`),
- **evaluating** densities and probabilities (`pdf`, `logpdf`, `cdf`),
- **summarising** distributions (`mean`, `var`, `params`), and
- **fitting** distributions to data (`fit`).

`ExponentialFamily.jl` focuses on the subset of distributions that belong to the
[exponential family](@ref exponential-family-primer) and adds the specialised
machinery that this shared structure makes possible:

- a **natural-parameter representation** via the generic
  [`ExponentialFamilyDistribution`](@ref) type,
- **analytic products** of distributions over the same variable (the core
  operation behind Bayes' rule and conjugate inference),
- direct access to **exponential family attributes** — base measure, sufficient
  statistics, log partition function, and
  [Fisher information](https://en.wikipedia.org/wiki/Fisher_information), and
- a number of **extra parameterizations and distributions** that are convenient
  for inference but are not part of `Distributions.jl` (see the
  [Library](@ref library-list-distributions-extra) page).

## At a glance

| | `Distributions.jl` | `ExponentialFamily.jl` |
|---|---|---|
| Scope | Distributions of all kinds | The exponential family subset |
| Parameterization | Mean parameters (e.g. mean, variance, probability) | Natural parameters ``\eta``, plus conversions to/from mean parameters |
| Sampling (`rand`) | ✔ Primary feature | Uses `Distributions.jl` |
| Density (`pdf`, `logpdf`) | ✔ | ✔ (also in natural form) |
| Fitting to data (`fit`) | ✔ | Uses `Distributions.jl` |
| Analytic product of distributions | — | ✔ |
| Sufficient statistics / log partition / Fisher information | — | ✔ |
| Relationship | Standalone foundation | Builds on top of `Distributions.jl` |

## Using both together

Because `ExponentialFamily.jl` re-uses the `Distributions.jl` types directly, you
move between the two worlds with a single `convert` call. Start from an ordinary
`Distributions.jl` distribution defined by its familiar mean parameters:

```@example comparison
using ExponentialFamily, Distributions

bernoulli = Bernoulli(0.25) # the usual Distributions.jl distribution
```

Convert it into its exponential family (natural parameter) representation when you
need the extra machinery:

```@example comparison
ef = convert(ExponentialFamilyDistribution, bernoulli)
```

And convert straight back to the `Distributions.jl` type when you are done:

```@example comparison
convert(Bernoulli, ef)
```

The round trip recovers the original distribution. In practice you will keep using
`Distributions.jl` for sampling and density evaluation, and reach for
`ExponentialFamily.jl` when you need products, sufficient statistics, or other
exponential family attributes.

## When should I use which?

- Reach for **`Distributions.jl`** when you want to draw samples, evaluate a
  density, fit a distribution to data, or work with a distribution that is not in
  the exponential family.
- Reach for **`ExponentialFamily.jl`** when you need to multiply distributions
  analytically (e.g. inside a Bayesian inference routine), or when you need the
  natural parameters, sufficient statistics, log partition function, or Fisher
  information of an exponential family member.

## Where to go next

- [The `ExponentialFamilyDistribution` Interface](@ref interface) — the full API.
- [Examples](@ref examples) — products and attribute computations in action.
- [Library](@ref library-list-distributions-extra) — the additional
  distributions provided by this package.
