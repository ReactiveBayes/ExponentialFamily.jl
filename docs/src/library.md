# [Library](@id library)

In the context of the package, exponential family distributions are represented in the form 
$$\begin{aligned}
f_X(x\mid\theta) = h(x)\,\exp\!\bigl[\,\eta(\theta) \cdot T(x) - A(\theta)\,\bigr]
\end{aligned}$$
where $h(x)$ - basemeasure, $T(x)$ - sufficient statistics, $A(\theta)$ - log partition, $\eta(\theta)$ - natural parameters.

## Main Structure
```@docs
ExponentialFamilyDistribution
```

## Methods

```@docs
ExponentialFamily.getlogpartition
ExponentialFamily.getbasemeasure
ExponentialFamily.getsufficientstatistics
ExponentialFamily.getnaturalparameters
ExponentialFamily.getsupport
ExponentialFamily.logpdf
ExponentialFamily.pdf
ExponentialFamily.fisherinformation
```

## Additional distributions
These are the distributions that are not included in the Distributions.jl package.
```@docs
ExponentialFamily.Contingency
ExponentialFamily.ContinuousBernoulli
ExponentialFamily.MatrixDirichlet
ExponentialFamily.GammaShapeRate
ExponentialFamily.GammaShapeScale
ExponentialFamily.NormalMeanPrecision
ExponentialFamily.NormalMeanVariance
ExponentialFamily.NormalWeightedMeanPrecision
ExponentialFamily.MvNormalMeanPrecision
ExponentialFamily.MvNormalMeanCovariance
ExponentialFamily.MvNormalWeightedMeanPrecision
ExponentialFamily.JointNormal
ExponentialFamily.WishartFast
ExponentialFamily.InverseWishartFast
```


## Prod related methods

```@docs
prod(::ClosedProd, left, right)
ExponentialFamily.ProdPreserveType
ExponentialFamily.ProdPreserveTypeLeft
ExponentialFamily.ProdGeneric
ExponentialFamily.ProdPreserveTypeRight
ExponentialFamily.closed_prod_rule
ExponentialFamily.ClosedProd
ExponentialFamily.LinearizedProductDistribution
ExponentialFamily.ProductDistribution
```