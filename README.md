# ExponentialFamily.jl

| **Documentation**                                                         | **Build Status**                                       | **Code Coverage**                       | 
|:-------------------------------------------------------------------------:|:------------------------------------------------------:|:---------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI][ci-img]][ci-url] [![Aqua][aqua-img]][aqua-url]  | [![CodeCov][codecov-img]][codecov-url]  |

[ci-img]: https://github.com/biaslab/ExponentialFamily.jl/actions/workflows/CI.yml/badge.svg?branch=main
[ci-url]: https://github.com/biaslab/ExponentialFamily.jl/actions

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://biaslab.github.io/ExponentialFamily.jl/dev

[codecov-img]: https://codecov.io/gh/biaslab/ExponentialFamily.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/biaslab/ExponentialFamily.jl?branch=main

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://biaslab.github.io/ExponentialFamily.jl/stable

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

ExponentialFamily.jl is a Julia package that extends the functionality of Distributions.jl by providing a collection of exponential family distributions and customized implementations. It is designed to facilitate working with exponential family distributions and offers specialized functionality tailored to this class of distributions.


## Key Features

- **Distributions**: ExponentialFamily.jl includes a wide range of exponential family distributions such as Gaussian, Gamma, Poisson, Fisher, and more. These distributions are built upon the foundation of Distributions.jl, ensuring compatibility and consistency.

- **Analytic Products**: The package provides support for analytic products of distributions belonging to the same exponential family. These products are useful for conjugate computations in inference algorithms, enabling efficient calculations in Bayesian inference and other probabilistic models.

- **Analytical attributes**: ExponentialFamily.jl offers hand-written and efficient functions to compute various attributes of members of the exponential family, such as base measure, sufficient statistics, the log partition function, mappings between different parametrizations and others.

- **Fisher Information**: ExponentialFamily.jl also offers computation of the [Fisher Information (FI)](https://en.wikipedia.org/wiki/Fisher_information) for various distributions. FI is a crucial quantity in statistical inference, providing insights into the sensitivity of a model's parameters to changes in the data. Essentially FI is the hessian of logpartition with respect to vectorized natural parameters. FI allows users to gain a deeper understanding of the behavior and performance of their probabilistic models.

Read more about the package in the [documentation](https://biaslab.github.io/ExponentialFamily.jl/stable/).

## Installation
ExponentialFamily.jl can be installed through the Julia package manager. In the Julia REPL, type `]` to enter the package manager mode and run:
```julia
pkg> add ExponentialFamily
```

# License

[MIT License](LICENSE) Copyright (c) 2023-2024 BIASlab
