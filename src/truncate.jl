export TruncatedExponentialFamilyDistribution
import Distributions:_in_closed_interval, logsubexp

struct TruncatedExponentialFamilyDistribution{D<:ExponentialFamilyDistribution{<:UnivariateDistribution}, S<:ValueSupport, T<: Real, TL<:Union{T,Nothing}, TU<:Union{T,Nothing}} <: UnivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::TL     # lower bound
    upper::TU     # upper bound
    loglcdf::T    # log-cdf of lower bound (exclusive): log P(X < lower)
    lcdf::T       # cdf of lower bound (exclusive): P(X < lower)
    ucdf::T       # cdf of upper bound (inclusive): P(X ≤ upper)

    tp::T         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::T      # log(tp), i.e. log(ucdf - lcdf)

    function TruncatedExponentialFamilyDistribution(d::ExponentialFamilyDistribution{<:UnivariateDistribution}, l::TL, u::TU, loglcdf::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real, TL <: Union{T,Nothing}, TU <: Union{T,Nothing}}
        new{typeof(d), value_support(typeof(d)), T, TL, TU}(d, l, u, loglcdf, lcdf, ucdf, tp, logtp)
    end
end

const LeftTruncatedExponentialFamilyDistribution{D<:ExponentialFamilyDistribution{<:UnivariateDistribution},S<:ValueSupport,T<:Real} = 
                        TruncatedExponentialFamilyDistribution{D,S,T,T,Nothing}
const RightTruncatedExponentialFamilyDistribution{D<:ExponentialFamilyDistribution{<:UnivariateDistribution},S<:ValueSupport,T<:Real} = 
                        TruncatedExponentialFamilyDistribution{D,S,T,Nothing,T}

function Distributions.truncated(d::TruncatedExponentialFamilyDistribution, l::T, u::T) where {T<:Real}
    return Distributions.truncated(
        d.untruncated,
        d.lower === nothing ? l : max(l, d.lower),
        d.upper === nothing ? u : min(u, d.upper),
    )
end
function Distributions.truncated(d::TruncatedExponentialFamilyDistribution, ::Nothing, u::Real)
    return Distributions.truncated(d.untruncated, d.lower, d.upper === nothing ? u : min(u, d.upper))
end
function Distributions.truncated(d::TruncatedExponentialFamilyDistribution, l::Real, ::Nothing)
    return Distributions.truncated(d.untruncated, d.lower === nothing ? l : max(l, d.lower), d.upper)
end

function Distributions.truncated(ef::ExponentialFamilyDistribution,l::T,u::T) where {T <: Real}
    loglcdf_ = log(cdf(ef, l))
    lcdf_    = cdf(ef, l)
    ucdf_    = cdf(ef, u)
    tp_      = ucdf_ - lcdf_
    logtp_   = log(tp_)
    return TruncatedExponentialFamilyDistribution(ef,l,u,loglcdf_,lcdf_,ucdf_,tp_,logtp_)
end
BayesBase.params(d::TruncatedExponentialFamilyDistribution) = tuple(params(d.untruncated)..., d.lower, d.upper)
Distributions.partype(d::TruncatedExponentialFamilyDistribution{<:UnivariateDistribution,<:ValueSupport,T}) where {T<:Real} = promote_type(partype(d.untruncated), T)

Base.eltype(::Type{<:TruncatedExponentialFamilyDistribution{D}}) where {D<:UnivariateDistribution} = eltype(D)
Base.eltype(d::TruncatedExponentialFamilyDistribution) = eltype(d.untruncated)

### range and support

islowerbounded(d::RightTruncatedExponentialFamilyDistribution) = islowerbounded(d.untruncated)
islowerbounded(d::TruncatedExponentialFamilyDistribution) = islowerbounded(d.untruncated) || isfinite(d.lower)

isupperbounded(d::LeftTruncatedExponentialFamilyDistribution) = isupperbounded(d.untruncated)
isupperbounded(d::TruncatedExponentialFamilyDistribution) = isupperbounded(d.untruncated) || isfinite(d.upper)

minimum(d::RightTruncatedExponentialFamilyDistribution) = minimum(d.untruncated)
minimum(d::TruncatedExponentialFamilyDistribution) = max(minimum(d.untruncated), d.lower)

maximum(d::LeftTruncatedExponentialFamilyDistribution) = maximum(d.untruncated)
maximum(d::TruncatedExponentialFamilyDistribution) = min(maximum(d.untruncated), d.upper)

function BayesBase.insupport(d::TruncatedExponentialFamilyDistribution{<:UnivariateDistribution,<:Union{Discrete,Continuous}}, x::Real)
    return _in_closed_interval(x, d.lower, d.upper) && insupport(d.untruncated, x)
end

### evaluation

function Distributions.quantile(d::TruncatedExponentialFamilyDistribution, p::Real)
    x = Distributions.quantile(d.untruncated, d.lcdf + p * d.tp)
    min_x, max_x = extrema(d)
    return clamp(x, oftype(x, min_x), oftype(x, max_x))
end

function BayesBase.pdf(d::TruncatedExponentialFamilyDistribution, x::Real)
    result = pdf(d.untruncated, x) / d.tp
    return _in_closed_interval(x, d.lower, d.upper) ? result : zero(result)
end

function BayesBase.logpdf(d::TruncatedExponentialFamilyDistribution, x::Real)
    result = logpdf(d.untruncated, x) - d.logtp
    return _in_closed_interval(x, d.lower, d.upper) ? result : oftype(result, -Inf)
end

function BayesBase.cdf(d::TruncatedExponentialFamilyDistribution, x::Real)
    result = (cdf(d.untruncated, x) - d.lcdf) / d.tp
    return if d.lower !== nothing && x < d.lower
        zero(result)
    elseif d.upper !== nothing && x >= d.upper
        one(result)
    else
        result
    end
end

function Distributions.logcdf(d::TruncatedExponentialFamilyDistribution, x::Real)
    result = logsubexp(logcdf(d.untruncated, x), d.loglcdf) - d.logtp
    return if d.lower !== nothing && x < d.lower
        oftype(result, -Inf)
    elseif d.upper !== nothing && x >= d.upper
        zero(result)
    else
        result
    end
end

function Distributions.ccdf(d::TruncatedExponentialFamilyDistribution, x::Real)
    result = (d.ucdf - cdf(d.untruncated, x)) / d.tp
    return if d.lower !== nothing && x <= d.lower
        one(result)
    elseif d.upper !== nothing && x > d.upper
        zero(result)
    else
        result
    end
end

function Distributions.logccdf(d::TruncatedExponentialFamilyDistribution, x::Real)
    result = logsubexp(logccdf(d.untruncated, x), log1p(-d.ucdf)) - d.logtp
    return if d.lower !== nothing && x <= d.lower
        zero(result)
    elseif d.upper !== nothing && x > d.upper
        oftype(result, -Inf)
    else
        result
    end
end

### Exponential Family related 

getlogpartition(::NaturalParametersSpace, tef::TruncatedExponentialFamilyDistribution) = (η) -> begin
    ef = tef.untruncated
    lp = getlogpartition(ef)(η)
    return log(tef.ucdf - tef.lcdf) - lp
end

getgradlogpartition(::NaturalParametersSpace, tef::TruncatedExponentialFamilyDistribution) = (η) -> begin
    ef = tef.untruncated
    c  = tef.ucdf - tef.lcdf
    T  = exponential_family_typetag(ef)
    space = NaturalParametersSpace()
    uf  = getgradcdf(space, T)(η, tef.upper)
    lf  = getgradcdf(space, T)(η, tef.lower)
    grad =  getgradlogpartition(ef)(η)
    return (uf - lf)/c - grad 
end

getfisherinformation(::NaturalParametersSpace, tef::TruncatedExponentialFamilyDistribution) = (η) -> begin
    ef = tef.untruncated
    T  = exponential_family_typetag(ef)
    c = tef.ucdf - tef.lcdf
    space = NaturalParametersSpace()
    uf  = getgradcdf(space, T)(η, tef.upper)
    lf  = getgradcdf(space, T)(η, tef.lower)
    hessu = gethessiancdf(space, T)(η, tef.upper)
    hessl = gethessiancdf(space, T)(η, tef.lower)

    return (hessu - hessl)/c - kron(uf -lf, (uf -lf)')/c^2 - getfisherinformation(ef)(η)
end




