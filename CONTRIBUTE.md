# Distributions style guide

### Methods' order
```julia
vague(::Type{<:Distribution})

# moments if necessary
function mean(::typeof(log), dist::ExponeDistributionntial)
    # ...
end

function check_valid_natural(::Type{<:Distribution}, params) 
    # ...
end

function isproper(::NaturalParameters{Distribution}, params)
    # ...
end

function lognormalizer(dist::Distribution)
    # ...
end

function check_valid_natural(::Type{<:Distribution}, params) 
    # ...
end

function isproper(::NaturalParameters{Distribution}, params) 
    # ...
end

function basemeasure(::Type{<:Distribution}, dist) 
    # ...
end

function basemeasure(::NaturalParameters{Distribution}, params) 
    # ...
end

function plus(::NaturalParameters{Distribution}, ::NaturalParameters{Distribution})
    # ...
end

prod_analytical_rule(::Type{<:Distribution}, ::Type{<:Distribution}) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdAnalytical, left::Distribution, right::Distribution)
    # ...
end

function Base.convert(::Type{NaturalParameters}, dist::Distribution)
    # ...
end

function Base.convert(::Type{Distribution}, params::NaturalParameters{Distribution})
    # ...
end

```