# Changelog

All notable changes to ExponentialFamily.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `MvNormalGamma` distribution — the multivariate generalization of `NormalGamma`, i.e. the joint conjugate prior over a Gaussian mean vector `θ` and a scalar precision `γ` (`θ ∣ γ ~ N(μ, (γΛ)⁻¹)`, `γ ~ Gamma(α, β)`). Includes the full `ExponentialFamily` interface (natural parameters, log-partition, gradient, Fisher information), `prod`, sampling, `logpdf`, and differential `entropy` ([#287](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/287)).
