# [Library](@id library)

## [Additional distributions](@id library-list-distributions-extra)

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
ExponentialFamily.default_prod_rule
ExponentialFamily.ClosedProd
ExponentialFamily.PreserveTypeProd
ExponentialFamily.PreserveTypeLeftProd
ExponentialFamily.PreserveTypeRightProd
ExponentialFamily.GenericProd
ExponentialFamily.ProductOf
ExponentialFamily.LinearizedProductOf
```