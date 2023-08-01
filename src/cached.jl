
struct NoCachedImplementation end

"""
    cache_for(operation, object)

Returns a cache structure for the efficient subsequent calls of the `operation` given specific `object`.
For example, `cache = cache_for(logpdf, MvNormal(0, 1))` should return a suitable 
cache structure for efficient execution of `logpdf(cache, Normal(0, 1), 0.0)`.
May return `NoCachedImplementation()`, in which case `operation!` call may fail with the `MethodError`.
Use `cached(operation, object)` to return an actual callable object, which fallbacks to the `operation` in 
case of `NoCachedImplementation()`.

Using cached version of specific functions, such as `logpdf` or `fisherinformation` may speed-up the computations 
significantly, especially within for loops.

```jldoctest
julia> distribution = MvNormal([ 0.0, 0.0 ], [ 1.0 0.0; 0.0 1.0 ]);

julia> cache = cache_for(logpdf, distribution);

julia> @allocated logpdf(cache, distribution, [ 0.0, 0.0 ])

julia> @allocated logpdf(cache, distribution, [ 1.0, 1.0 ]) # subsequent calls are faster and allocate less, due to internal caching
```

!!! note: Incorrect usage of cached operations (e.g size or dimension incompatibility) may (and probably will) 
          lead to incorrect results, memory corruptions and hard to track bugs, use at your own risk.

See also: [`cached`](@ref), [`NoCacheAvailable`](@ref)
"""
cache_for(_, __) = NoCachedImplementation()

cached(operation, object) = cached(cache_for(operation, object), operation, object)

# Fallback to the same `operation` in case no efficient implementation is available
cached(::NoCachedImplementation, operation, _) = operation

# Return the infused version of the `operation` in case cache is not `NoCachedImplementation`
cached(cache, operation, _) = let cache = cache
    (args...) -> operation(cache, args...)
end