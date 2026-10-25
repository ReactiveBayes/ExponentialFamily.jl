# [Examples](@id examples)

## Fisher-information–vector products

Use [`fishermul`](@ref) to apply the Fisher information in natural parameter space.
For scalable built-in families this avoids constructing the full Fisher matrix,
including when that matrix is singular, as for Categorical distributions.

```@example fishermul
using ExponentialFamily, Distributions

q = convert(ExponentialFamilyDistribution, Dirichlet([2.0, 3.0, 4.0]))
v = [1.0, -2.0, 0.5]
fishermul(q, v) ≈ fisherinformation(q) * v
```

The explicit parameter space is the first argument; omitting it selects
`NaturalParametersSpace()`. `DefaultParametersSpace()` is not implemented yet.
An explicit parameter vector can be supplied before the direction:

```@example fishermul
space = NaturalParametersSpace()
η = getnaturalparameters(q)
f = getfishermul(space, q)
f(η, v) ≈ fishermul(space, q, η, v)
```

For repeated calls, allocate the destination and workspace once:

```@example fishermul
w = similar(v, float(promote_type(eltype(η), eltype(v))))
buffer = FisherMulBuffer(space, q, v)
fishermul!(w, space, q, v, buffer)
w ≈ fishermul(q, v)
```

[`fishermul!`](@ref) overwrites and returns `w`. The destination must not overlap
the parameters or direction. Calls without a buffer create temporary workspace
internally when needed. A compatible buffer eliminates heap allocations in
warmed-up calls for built-in families with valid dense Float32/Float64 arrays. Generic
custom-family fallbacks and other numeric types do not carry that guarantee.

The buffer can be reused when parameter and direction values change, provided
their dimensions and promoted numeric type, family, space, and conditioner stay
compatible. Parameter-dependent workspace is refreshed on every call. Each
concurrent call needs a separate buffer. The callable returned by
`getfishermul!(space, q)` accepts `(w, η, v, buffer = nothing)` and holds no mutable
cache. The default allocating wrapper calls it with the buffer omitted.

### Using fishermul to compute the natural gradient

The natural gradient is a popular technique in optimization and variational inference.

The natural gradient can be understood as preconditioning the ordinary gradient with the inverse of the Fisher information matrix, which accounts for the geometry of the parameter space. You can read more about it in [Information Geometry of Variational Bayes](https://arxiv.org/abs/2509.15641) by Mohammad Emtiyaz Khan (2025).

Given a Euclidean gradient ``v`` with respect to the natural parameters ``\eta``, the natural gradient is ``F(\eta)^{-1}v``, where ``F(\eta)`` is the Fisher information matrix at ``\eta``.

For example, for the Dirichlet distribution, we can use `fishermul!` with conjugate gradients to compute the natural gradient without explicitly inverting the Fisher information matrix. Below, we use a random vector `v` as an example Euclidean gradient in 10 dimensions and compare the results and timings of both methods.

```@example fishermul
using LinearAlgebra, Random
using Krylov, LinearOperators
using FastCholesky, BenchmarkTools

d = 10
v = randn(MersenneTwister(42), d)
ef = convert(ExponentialFamilyDistribution, Dirichlet(ones(d)))
Fv = LinearOperator(
    Float64, d, d, true, true,
    (y, x) -> fishermul!(y, NaturalParametersSpace(), ef, x)
)

# Check the result and warm up both methods before timing.
F = fisherinformation(ef)
g_dense = cholinv(F) * v
g, stats = cg(Fv, v; atol = 1e-10, rtol = 1e-8, itmax = 200)
@assert stats.solved
@assert isapprox(g, g_dense; atol = 1e-10, rtol = 1e-8)
@assert norm(F * g - v) <= 1e-10 + 1e-8 * norm(v)

dense_time = @belapsed cholinv(fisherinformation($ef)) * $v seconds = 1
cg_time = @belapsed cg($Fv, $v; atol = 1e-10, rtol = 1e-8, itmax = 200) seconds = 1

(dense_ns = dense_time * 1e9, cg_ns = cg_time * 1e9)
```

If you are interested in benchmarks across different dimensions, see the plot and full example in [PR #326](https://github.com/ReactiveBayes/ExponentialFamily.jl/pull/326).

## [Product of two probability distributions over the same variable](@id examples-product)

In this example, we'll show you how to use the ExponentialFamily package to calculate the product of probability distributions that both relate to the same variable "X". 
This operation results in another probability distribution, also centered around the variable "X". 
It's essential to note that this is distinct from multiplying probability distributions for two different variables, such as "X" and "Y", 
which yields a joint probability distribution. Calculating the product of two probability distributions over the same variable is a crucial step in applying Bayes' rule.

```math 
p(X\vert D) \propto \underbrace{p(X)p(D|X)}_{\mathrm{product~of~two~distributions}}
```

To perform this operation, the ExponentialFamily library employs the `prod` function. This function takes a product strategy as its first argument. For instance:

```@example prod-example
using ExponentialFamily, Distributions, BayesBase

prior = Bernoulli(0.5)
likelihood = Bernoulli(0.6)

posterior = prod(PreserveTypeProd(Distribution), prior, likelihood)
```

You can achieve the same result in the general exponential family form, which operates within the natural parameter space:

```@example prod-example

ef_prior = convert(ExponentialFamilyDistribution, prior)
ef_likelihood = convert(ExponentialFamilyDistribution, likelihood)

ef_posterior = prod(PreserveTypeProd(ExponentialFamilyDistribution), ef_prior, ef_likelihood)
```

Or even more concisely:

```@example prod-example
prod(PreserveTypeProd(ExponentialFamilyDistribution), prior, likelihood)
```

In this example, multiplying two `Bernoulli` distributions will always result in another `Bernoulli` distribution. However, this is not the case for all distributions. For instance, the product of two `Laplace` distributions may not yield another `Laplace` distribution, and representing the result in the same form might not be possible. In such cases, it's advisable to calculate the result within the exponential family domain. This is because the product of two exponential family distributions can always be represented as another exponential family distribution, as shown here:

```@example prod-example

prior = Laplace(2.0, 3.0)
likelihood = Laplace(1.0, 4.0)

prod(PreserveTypeProd(ExponentialFamilyDistribution), prior, likelihood)
```

Note that the result does not correspond to the `Laplace` distribution and returns a generic univariate `ExponentialFamilyDistribution`.
This approach ensures consistency and compatibility, especially when dealing with a wide range of probability distributions.
Refer to the [`BayesBase`](https://github.com/ReactiveBayes/BayesBase.jl) for the documentation about available product strategies.

### [Multinomial products](@id examples-multinomial-product)

Two `Multinomial` distributions over the same count vector must have the same
trial count and number of categories. Their normalized product is a generic
multivariate `ExponentialFamilyDistribution`, with a squared multinomial base
measure. For example:

```@example multinomial-product
using ExponentialFamily, Distributions, BayesBase

left = Multinomial(5, [0.2, 0.3, 0.5])
right = Multinomial(5, [0.4, 0.4, 0.2])
product = prod(PreserveTypeProd(ExponentialFamilyDistribution), left, right)
pdf(product, [1, 2, 2])
```

Let ``\eta`` be the sum of the two natural-parameter vectors. The product's
normalizer is

```math
Z_n(\eta) = \sum_{\substack{x_1+\cdots+x_K=n\\x_i\geq 0}}
    \left(\frac{n!}{x_1!\cdots x_K!}\right)^2
    \exp\!\left(\sum_{i=1}^K \eta_i x_i\right),
\qquad A(\eta)=\log Z_n(\eta).
```

This is the **Conway–Maxwell–multinomial** family with dispersion ``\nu=2``.
Morris, Raim and Sellers (2020), §3, equations (3)–(4) in the
[preprint](https://arxiv.org/pdf/1911.02131), define its PMF and normalizer
``C(p,\nu;n)``. Their probability parameter is ``p=\operatorname{softmax}(\eta)``,
so ``Z_n(\eta)=(\sum_i e^{\eta_i})^n C(p,2;n)``. For input probability vectors
``a`` and ``b``, the product therefore has ``p_i=a_i b_i/\sum_j a_j b_j``.

#### Exact evaluation by convolution

The following recurrence is derived from that normalizer; it is not an algorithm
claimed by the cited paper. Write ``Z_j(m)`` for the normalizer with the first
``j`` categories and ``m`` trials. Factoring the multinomial coefficient as

```math
\binom{m}{x_1,\ldots,x_{j-1},r}
=\binom mr\binom{m-r}{x_1,\ldots,x_{j-1}}
```

gives

```math
Z_1(m)=e^{m\eta_1},\qquad
Z_j(m)=\sum_{r=0}^{m}\binom mr^2 e^{r\eta_j}Z_{j-1}(m-r).
```

The implementation evaluates ``L_j(m)=\log Z_j(m)`` with `logaddexp` and
log-binomial coefficients computed from `loggamma`. It returns ``L_K(n)`` using
``O(Kn^2)`` time and ``O(n)`` workspace. This sums the full support up to floating
point rounding, with no random sampling or truncation. Workspace is private to
each evaluation, and derivatives of the product log-partition are available
through ForwardDiff. Categories with zero weight are omitted from the convolution;
a product with no positive mass is rejected. With zero trials, the sole outcome
is the all-zero count vector and its mass is one.

There is also a special-function identity, obtained by matching coefficients in
the definition of Lauricella's ``F_C`` in Srivastava and Exton (1974), equation (2):

```math
Z_n(\eta)=e^{n\eta_K}
F_C^{(K-1)}\!\left(-n,-n;1,\ldots,1;
e^{\eta_1-\eta_K},\ldots,e^{\eta_{K-1}-\eta_K}\right).
```

For finite ``\eta`` and ``K\geq2``, this series terminates because its first
two parameters are ``-n``. For ``K=2`` it reduces to
``e^{n\eta_2}{}_2F_1(-n,-n;1;e^{\eta_1-\eta_2})``. The implementation uses the
convolution recurrence, so no multivariate special-function dependency is needed.

#### Parameter conventions and Fisher information

The ordinary Multinomial interface retains all ``K`` log weights. For positive
probabilities, conversion uses ``\eta_i=\log p_i-\log p_K``; when ``p_K=0``,
the last positive category is used as the reference. Zero probabilities correspond
to ``-\infty`` log weights, with zero counts contributing zero to the log density.

The natural-space Fisher matrix is ``n(\operatorname{Diag}(p)-pp^\mathsf{T})``.
For positive probabilities and ``n>0``, it has rank ``K-1`` and null direction
``\boldsymbol{1}``. The default-space matrix, on normalized probabilities, is
``n(\operatorname{Diag}(1/p)-\boldsymbol{1}\boldsymbol{1}^\mathsf{T})`` and has
null direction ``p``. These are singular because the coordinates are redundant;
inversion requires restricting to the identifiable subspace. Default-space Fisher
evaluation requires strictly positive probabilities. `fishermul!` applies the
natural-space matrix without constructing it and supports reusable buffers.

The default-space log-partition retains the term ``n\log(\sum_i p_i)`` even
though it vanishes on the simplex. This ensures its derivatives agree with the
log-ratio parameter mapping when coordinates are differentiated independently.

References:

- D. S. Morris, A. M. Raim and K. F. Sellers (2020), *A Conway–Maxwell–multinomial
  distribution for flexible modeling of clustered categorical data*, Journal of
  Multivariate Analysis **179**, 104651.
  [DOI: 10.1016/j.jmva.2020.104651](https://doi.org/10.1016/j.jmva.2020.104651).
- H. M. Srivastava and H. Exton (1974), *A transformation formula relating two
  Lauricella functions*, Rendiconti dell'Accademia Nazionale dei Lincei,
  Classe di Scienze Fisiche, Matematiche e Naturali, series 8, **56**(1), 38–42.
  [Article, equation (2)](https://www.bdim.eu/item?fmt=pdf&id=RLINA_1974_8_56_1_38_0).

```@docs
ExponentialFamily.__compute_logpartition_multinomial_product
```

## Computing various useful attributes of an exponential family member

The package implements attributes of many well known exponential family members, which are defined in [this table](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions).
The attributes include [`getbasemeasure`](@ref), [`getsufficientstatistics`](@ref), [`getlogpartition`](@ref), [`getfisherinformation`](@ref), and others. 
In general, the interface for these functions assumes a family member "tag," such as `Normal` or `Bernoulli`. Here are some examples of how to use these attributes:

```@example attributes-example
using ExponentialFamily, Distributions

# Returns a function
basemeasure_of_normal = getbasemeasure(Normal)

basemeasure_of_normal(0.0)
```

```@example attributes-example

# Returns an iterable of functions
sufficientstatistics_of_gamma = getsufficientstatistics(Gamma)

map(f -> f(1.0), sufficientstatistics_of_gamma)
```

Some distributions, like the `Laplace` distribution, qualify as exponential family members only under certain conditions or when specific information is known in advance. In such cases, the ExponentialFamily package introduces the concept of a __conditioner__. For instance, the `Laplace` distribution becomes a member of the exponential family only when the `location` parameter is known and fixed. Consequently, we __condition__ on the `location` parameter:

```@example attributes-example
laplace = Laplace(2.0, 1.0)

canonical = convert(ExponentialFamilyDistribution, laplace)

getconditioner(canonical)
```

```@example attributes-example
# For conditioned distributions, the `conditioner` must be present as a second argument
basemeasure_of_laplace = getbasemeasure(Laplace, 2.0)

basemeasure_of_laplace(1.0)
```

The [`getlogpartition`](@ref) and [`getfisherinformation`](@ref) functions optionally accept a `space` parameter as the first argument. This `space` parameter specifies the parameterization `space`, such as [`DefaultParametersSpace`](@ref) or [`NaturalParametersSpace`](@ref). The result obtained from these functions (in general) depends on the chosen parameter space:

```@example attributes-example
logpartition_of_gamma_in_mean_space = getlogpartition(DefaultParametersSpace(), Gamma)

gamma_parameters_in_mean_space = [ 1.0, 2.0 ]

logpartition_of_gamma_in_mean_space(gamma_parameters_in_mean_space)
```

```@example attributes-example
logpartition_of_gamma_in_natural_space = getlogpartition(NaturalParametersSpace(), Gamma)

gamma_parameters_in_natural_space = map(
    DefaultParametersSpace() => NaturalParametersSpace(), 
    Gamma,
    gamma_parameters_in_mean_space
)

logpartition_of_gamma_in_natural_space(gamma_parameters_in_natural_space)
```

The same principle applies to the Fisher information matrix:

```@example attributes-example
fisherinformation_of_gamma_in_mean_space = getfisherinformation(DefaultParametersSpace(), Gamma)

fisherinformation_of_gamma_in_mean_space(gamma_parameters_in_mean_space)
```

```@example attributes-example
fisherinformation_of_gamma_in_natural_space = getfisherinformation(NaturalParametersSpace(), Gamma)

fisherinformation_of_gamma_in_natural_space(gamma_parameters_in_natural_space)
```

## Approximating attributes 

Refer to the [`ExpectationApproximations.jl`](https://github.com/ReactiveBayes/ExpectationApproximations.jl) package for approximating various attributes of the members of the exponential family.
