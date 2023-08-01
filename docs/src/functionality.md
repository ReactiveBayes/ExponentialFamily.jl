
## Extension methods

In the context of the package, exponential family distributions are represented in the form 
$$\begin{aligned}
f_X(x\mid\theta) = h(x)\,\exp\!\bigl[\,\eta(\theta) \cdot T(x) - A(\theta)\,\bigr]
\end{aligned}$$
where $h(x)$ - basemeasure, $T(x)$ - sufficient statistics, $A(\theta)$ - log partition, $\eta(\theta)$ - natural parameters.

```@docs
logpartition
basemeasure
sufficientstatistics
fisherinformation
```

## Prod related methods

```@docs
prod_analytical_rule
resolve_prod_constraint
prod(::ProdAnalytical, left, right)
ProdAnalytical
ProdFinal
ProdPreserveType
ProdPreserveTypeLeft
ProdPr
DistProduct
ProdGeneric
GenericLogPdfVectorisedProduct
```