# Changelog

All notable changes to ExponentialFamily.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.6.2]

### Fixed
- Factorize `-η₂` rather than `η₂` in the `WishartFast` `getgradlogpartition` and `getfisherinformation`. The Wishart natural parameter `η₂` is negative definite and has no Cholesky factorization; the previous code produced correct values only by relying on an undocumented `FastCholesky` fallback that happened to factorize `-η₂`, and broke under alternative Cholesky implementations ([#321](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/321)).

### Changed
- Simplify the `VonMisesFisher` natural-space `getfisherinformation` to a single Bessel ratio, making it faster than the `ForwardDiff` baseline ([#321](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/321)).
- Use `@belapsed` instead of `@elapsed` in the distribution test harness timing comparisons for more reliable measurements.

## [2.6.1]

### Fixed
- Fix `Gamma` default-space `getgradlogpartition` returning both components with flipped signs ([#315](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/315)).
- Fix `NormalMeanVariance` default-space `getgradlogpartition` returning `1/σ²` instead of `1/(2σ²)` in its second component ([#315](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/315)).
- Fix `LogNormal` default-space `getgradlogpartition` using `abs(μ)`, which gave the wrong sign for negative `μ` ([#315](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/315)).
- Implement `quantile` for the custom univariate `GammaShapeRate` and Normal (`NormalMeanVariance`, `NormalMeanPrecision`, `NormalWeightedMeanPrecision`) parametrizations, and generically for any `ExponentialFamilyDistribution` backed by a `Distribution`. All of these previously threw a `MethodError` about `iterate` ([#268](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/268)).

### Added
- Add a generic `getgradlogpartition` check in `DefaultParametersSpace` to the distribution test harness, cross-validated against the natural space via the jacobian of the parameter mapping and against `ForwardDiff` ([#315](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/315)).
- Test against Julia `1.13` in CI.
- Implement `cdf` for `GammaShapeRate`, which had no method, for symmetry with the other univariate parametrizations.

### Changed
- Allow selecting a subset of test files from the command line via `make test test_args="..."`. `Aqua` checks are skipped for such runs unless `RUN_AQUA=true` is set ([#266](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/266)).
- Remove stale, non-discoverable work-in-progress test files under `test/distributions/wip/` ([#259](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/259)).

## [2.6.0]

### Fixed
- Fix default-space log-partition functions for `Erlang` and normal distributions, with regression tests ensuring consistency with natural-space values ([#316](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/316)).
- Fix `NegativeBinomial` default-space helpers: `isproper` return type, a `getgradlogpartition` crash, and a log-partition mismatch with natural space ([#291](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/291)).
- Fix `Weibull` default-space `getlogpartition`/`getgradlogpartition` (wrong value and shape, and a crash) ([#292](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/292)).
- Fix `DirichletCollection` default-space `getlogpartition` returning `0` instead of the per-slice partition sum ([#300](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/300)).
- Fix `InverseWishartFast` `getlogpartition` in `DefaultParametersSpace`.
- Fix `Geometric` default-space `getgradlogpartition` returning the wrong derivative ([#293](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/293)).
- Normalize products of two `NegativeBinomial` distributions over their full infinite support ([#299](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/299)).
- Fix `Categorical` natural-space gradient and Fisher information becoming `NaN` for large natural parameters; compute both via `softmax` ([#294](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/294)).
- Implement `logpdf`/`pdf`/`cdf` for `TruncatedExponentialFamilyDistribution`, which previously threw a `MethodError`; also remove an unused import ([#290](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/290)).
- Fix `Binomial` natural-space `getgradlogpartition` returning `NaN` for large logits; use `logistic` ([#297](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/297)).
- Fix `Bernoulli` natural-space `getlogpartition` overflowing to `Inf` for large logits; use `log1pexp` ([#298](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/298)).
- Fix `InverseWishartFast` default-space `getlogpartition` using `mvtrigamma` instead of `logmvgamma` ([#295](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/295)).

## [2.5.1]

### Fixed
- Resolve `convert` method ambiguity for `FullNormal` under Distributions 0.25.129 ([#288](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/288)).

### Changed
- Add beginner guides and fix spelling across documentation and docstrings ([#289](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/289)).

## [2.5.0]

### Added
- `MvNormalGamma` distribution — the multivariate generalization of `NormalGamma`, i.e. the joint conjugate prior over a Gaussian mean vector `θ` and a scalar precision `γ` (`θ ∣ γ ~ N(μ, (γΛ)⁻¹)`, `γ ~ Gamma(α, β)`). Includes the full `ExponentialFamily` interface (natural parameters, log-partition, gradient, Fisher information), `prod`, sampling, `logpdf`, and differential `entropy` ([#287](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/287)).
