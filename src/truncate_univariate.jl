export TruncatedExponentialFamilyDistribution
import Distributions:_in_closed_interval, logsubexp

struct TruncatedExponentialFamilyDistribution{D<:ExponentialFamilyDistribution{<:UnivariateDistribution}, S<:ValueSupport, T<: Real, TL<:Union{T,Nothing}, TU<:Union{T,Nothing}} <: UnivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::TL     # lower bound
    upper::TU     # upper bound
    lcdf::T       # cdf of lower bound (exclusive): P(X < lower)
    ucdf::T       # cdf of upper bound (inclusive): P(X ≤ upper)
    
    #constructor
    function TruncatedExponentialFamilyDistribution(d::ExponentialFamilyDistribution{<:UnivariateDistribution}, l::TL, u::TU, lcdf::T, ucdf::T) where {T <: Real, TL <: Union{T,Nothing}, TU <: Union{T,Nothing}}
        new{typeof(d), value_support(typeof(d)), T, TL, TU}(d, l, u, lcdf, ucdf)
    end

    #constructor with only bounds (Reals)
    function TruncatedExponentialFamilyDistribution(d::ExponentialFamilyDistribution{<:UnivariateDistribution}, 
                                                l::TL, u::TU) where {TL<:AbstractFloat, TU<:AbstractFloat}
        T = paramfloattype(d)   # Float64 is default if eltype(d) is Int, ensures Real
        lcdf = l === nothing ? zero(T) : T(cdf(d, l))
        ucdf = u === nothing ? one(T)  : T(cdf(d, u))
        new{typeof(d), value_support(typeof(d)), T, TL, TU}(d, l, u, lcdf, ucdf)
    end

    function TruncatedExponentialFamilyDistribution(d::ExponentialFamilyDistribution{<:UnivariateDistribution}, 
                                                l::TL, u::TU) where {TL<:Any, TU<:Any} 
        T = paramfloattype(d)
        return TruncatedExponentialFamilyDistribution(d, float(something(l, zero(T))), float(something(u, one(T))))
    end
end

#Left-truncated: special instance with upper_bound Nothing (infinity)
const LeftTruncatedExponentialFamilyDistribution{D<:ExponentialFamilyDistribution{<:UnivariateDistribution},S<:ValueSupport,T<:Real} = 
                        TruncatedExponentialFamilyDistribution{D,S,T,T,Nothing}
#Right-truncated: special instance with lower_bound Nothing (infinity)
const RightTruncatedExponentialFamilyDistribution{D<:ExponentialFamilyDistribution{<:UnivariateDistribution},S<:ValueSupport,T<:Real} = 
                        TruncatedExponentialFamilyDistribution{D,S,T,Nothing,T}

#Truncate the truncated distribution using the Julia Distributions package
function Distributions.truncated(d::TruncatedExponentialFamilyDistribution, l::T, u::T) where {T<:Real}
    return Distributions.truncated(
        d.untruncated,
        d.lower === nothing ? l : max(l, d.lower),
        d.upper === nothing ? u : min(u, d.upper),
    )
end

#functionality for right-truncated distributions
function Distributions.truncated(d::TruncatedExponentialFamilyDistribution, ::Nothing, u::Real)
    return Distributions.truncated(d.untruncated, d.lower, d.upper === nothing ? u : min(u, d.upper))
end

#functionality for left-truncated distributions
function Distributions.truncated(d::TruncatedExponentialFamilyDistribution, l::Real, ::Nothing)
    return Distributions.truncated(d.untruncated, d.lower === nothing ? l : max(l, d.lower), d.upper)
end

#parameters are the original parameters plus lower and upper bound
BayesBase.params(d::TruncatedExponentialFamilyDistribution) = tuple(params(d.untruncated)..., d.lower, d.upper)

#ensures consistent precision between base parameters and truncated distribution parameters (including bounds)
Distributions.partype(d::TruncatedExponentialFamilyDistribution{<:UnivariateDistribution,<:ValueSupport,T}) where {T<:Real} = promote_type(partype(d.untruncated), T)

# Define the element type (eltype) for truncated distributions.
# For the type version, inherit the element type from the underlying untruncated distribution type D.
Base.eltype(::Type{<:TruncatedExponentialFamilyDistribution{D}}) where {D<:UnivariateDistribution} = eltype(D)
# For an instance, delegate to the eltype of the stored untruncated distribution.
Base.eltype(d::TruncatedExponentialFamilyDistribution) = eltype(d.untruncated)

### Functionality for Range and Support

# Determine whether a (possibly truncated) distribution has a finite lower bound.
# For a right-truncated distribution, inherit the lower-boundedness from the untruncated base.
islowerbounded(d::RightTruncatedExponentialFamilyDistribution) = islowerbounded(d.untruncated)
# For a general truncated distribution, it is lower-bounded if either the base distribution is,
# or an explicit finite lower truncation limit is set.
islowerbounded(d::TruncatedExponentialFamilyDistribution) = islowerbounded(d.untruncated) || isfinite(d.lower)

# Determine whether a (possibly truncated) distribution has a finite upper bound.
# For a left-truncated distribution, inherit the upper-boundedness from the untruncated base.
isupperbounded(d::LeftTruncatedExponentialFamilyDistribution) = isupperbounded(d.untruncated)
# For a general truncated distribution, it is upper-bounded if either the base distribution is,
# or an explicit finite upper truncation limit is set.
isupperbounded(d::TruncatedExponentialFamilyDistribution) = isupperbounded(d.untruncated) || isfinite(d.upper)

# Return the minimum value of a (possibly truncated) distribution.
# For a right-truncated distribution, the minimum is the same as the untruncated distribution.
minimum(d::RightTruncatedExponentialFamilyDistribution) = minimum(d.untruncated)
# For a general truncated distribution, the minimum is the larger of the untruncated minimum
# and the lower truncation bound (if finite).
minimum(d::TruncatedExponentialFamilyDistribution) = max(minimum(d.untruncated), d.lower)

# Return the maximum value of a (possibly truncated) distribution.
# For a left-truncated distribution, the maximum is the same as the untruncated distribution.
maximum(d::LeftTruncatedExponentialFamilyDistribution) = maximum(d.untruncated)
# For a general truncated distribution, the maximum is the smaller of the untruncated maximum
# and the upper truncation bound (if finite).
maximum(d::TruncatedExponentialFamilyDistribution) = min(maximum(d.untruncated), d.upper)

# Check if a value `x` is within the support of a truncated distribution.
function BayesBase.insupport(d::TruncatedExponentialFamilyDistribution{<:UnivariateDistribution,<:Union{Discrete,Continuous}}, x::Real)
    return _in_closed_interval(x, d.lower, d.upper) && insupport(d.untruncated, x)
end


# function BayesBase.rand(rng::AbstractRNG, d::TruncatedExponentialFamilyDistribution)
#     d0 = convert(Distribution, d.untruncated)
#     tp = d.tp
#     lower = d.lower
#     upper = d.upper
#     if tp > 0.25
#         while true
#             r = rand(rng, d0)
#             if _in_closed_interval(r, lower, upper)
#                 return r
#             end
#         end
#     elseif tp > sqrt(eps(typeof(float(tp))))
#         return quantile(d0, d.lcdf + rand(rng) * d.tp)
#     else
#         return invlogcdf(d0, logaddexp(d.loglcdf, d.logtp - randexp(rng)))
#     end
# end