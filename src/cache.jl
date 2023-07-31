
"""
    CachedOperation(operation, cache)

!!! note: Do not create instances of `CachedOperation` directly, use the `cached` function instead.

See also: [`cached`](@ref), [`NoCacheAvailable`](@ref)
"""
struct CachedOperation{O, C}
    operation::O
    cache::C
end

"""
    NoCacheAvailable(operation)
"""
struct NoCacheAvailable{O}
    operation::O
end

"""
    cache(operation, object)

Returns a version of the `operation` specifically optimized for subsequent calls.
For example, `cached_f = cached(f, object)` should return a callable struct, which can be used to speed-up 
consequence calls to the `f` of the `object`. This is particularly useful to speedup common operations like, `logpdf` or 
`fisherinformation`, e.g.

```jldoctest
julia> distribution = MvNormal([ 0.0, 0.0 ], [ 1.0 0.0; 0.0 1.0 ]);

julia> cached_logpdf = cached(logpdf, distribution);

julia> cached_logpdf(distribution, [ 0.0, 0.0 ])

julia> cached_logpdf(distribution, [ 1.0, 1.0 ]) # subsequent calls are faster and allocate less, due to internal caching
```

!!! note: Incorrect usage of cached operations (e.g size or dimension incompatibility) may (and probably will) 
          lead to incorrect results, memory corruptions and hard to track bugs, use at your own risk.

See also: [`cached`](@ref), [`NoCacheAvailable`](@ref)
"""
cached(operation, object) = NoCacheAvailable(operation)