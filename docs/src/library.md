# [Library API](@id library)

## [Product API](@id library-prod)

One of the central functions in this package is the ability to calculate the product of two distributions over the same variable. 
You can also refer to the corresponding [example](@ref examples-product) for practical usage.

The `prod` function is a key feature of this package. 
It accepts a strategy as its first argument, which defines how the prod function should behave and what results you can expect.

```@docs
prod(::ClosedProd, left, right)
ExponentialFamily.default_prod_rule
```

### [Product strategies](@id library-prod-strategies)

For certain distributions, it's possible to compute the product using a straightforward mathematical equation, yielding a closed-form solution. 
However, for some distributions, finding a closed-form solution might not be feasible. 
Various strategies ensure consistent behavior in these situations. 
These strategies can either guarantee a fast and closed-form solution or, when necessary, fall back to a slower but more generic method.

```@docs
ExponentialFamily.UnspecifiedProd
ExponentialFamily.ClosedProd
ExponentialFamily.PreserveTypeProd
ExponentialFamily.PreserveTypeLeftProd
ExponentialFamily.PreserveTypeRightProd
ExponentialFamily.GenericProd
ExponentialFamily.ProductOf
ExponentialFamily.LinearizedProductOf
```

These strategies offer flexibility and reliability when working with different types of distributions, ensuring that the package can handle a wide range of cases effectively.

## [Additional distributions](@id library-list-distributions-extra)

These are the distributions that are not included in the Distributions.jl package.

```@docs
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
ExponentialFamily.JointGaussian
ExponentialFamily.NormalGamma
ExponentialFamily.MvNormalWishart
ExponentialFamily.FactorizedJoint
```

## [Promotion type utilities](@id library-promotion-utilities)

```@docs
ExponentialFamily.paramfloattype
ExponentialFamily.sampletype
ExponentialFamily.samplefloattype
ExponentialFamily.promote_variate_type
ExponentialFamily.promote_paramfloattype
ExponentialFamily.promote_sampletype
ExponentialFamily.promote_samplefloattype
ExponentialFamily.convert_paramfloattype
```

## [Extra stats functions](@id library-statsfuns)

```@docs
ExponentialFamily.mirrorlog
ExponentialFamily.xtlog
ExponentialFamily.logmvbeta
ExponentialFamily.clamplog
ExponentialFamily.mvtrigamma
```

## [Helper utilities](@id library-helpers)

```@docs
ExponentialFamily.vague
ExponentialFamily.logpdf_sample_optimized
ExponentialFamily.fuse_supports
```