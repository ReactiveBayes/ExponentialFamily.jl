# [Functionality](@id functionality)

## Main Structure
```@docs
ExponentialFamilyDistribution
```

## Methods

In the context of the package, exponential family distributions are represented in the form 
$$\begin{aligned}
f_X(x\mid\theta) = h(x)\,\exp\!\bigl[\,\eta(\theta) \cdot T(x) - A(\theta)\,\bigr]
\end{aligned}$$
where $h(x)$ - basemeasure, $T(x)$ - sufficient statistics, $A(\theta)$ - log partition, $\eta(\theta)$ - natural parameters.

```@docs
ExponentialFamily.logpartition
ExponentialFamily.basemeasure
ExponentialFamily.sufficientstatistics
ExponentialFamily.fisherinformation
```

## Prod related methods

```@docs
prod(::ClosedProd, left, right)
ProdPreserveType
ProdPreserveTypeLeft
ProdGeneric
```