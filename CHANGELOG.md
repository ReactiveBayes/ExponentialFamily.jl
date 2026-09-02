# Changelog

All notable changes to ExponentialFamily.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Implement `quantile` for the custom univariate `GammaShapeRate` and Normal (`NormalMeanVariance`, `NormalMeanPrecision`, `NormalWeightedMeanPrecision`) parametrizations, which previously threw a `MethodError` about `iterate` ([#268](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/268)).

### Changed
- Allow selecting a subset of test files from the command line via `make test test_args="..."` ([#266](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/266)).
- Remove stale, non-discoverable work-in-progress test files under `test/distributions/wip/` ([#259](https://github.com/ReactiveBayes/ExponentialFamily.jl/issues/259)).

## [2.5.1]

### Fixed
- Resolve `convert` method ambiguity for `FullNormal` under Distributions 0.25.129 ([#288](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/288)).

### Changed
- Add beginner guides and fix spelling across documentation and docstrings ([#289](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/289)).

## [2.5.0]

### Added
- `MvNormalGamma` distribution — the multivariate generalization of `NormalGamma`, i.e. the joint conjugate prior over a Gaussian mean vector `θ` and a scalar precision `γ` (`θ ∣ γ ~ N(μ, (γΛ)⁻¹)`, `γ ~ Gamma(α, β)`). Includes the full `ExponentialFamily` interface (natural parameters, log-partition, gradient, Fisher information), `prod`, sampling, `logpdf`, and differential `entropy` ([#287](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/287)).
