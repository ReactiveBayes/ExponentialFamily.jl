import QuadGK: quadgk

const GammaDistributionsFamily{T} = Union{GammaShapeScale{T}, GammaShapeRate{T}}

BayesBase.cov(dist::GammaDistributionsFamily) = var(dist)

function Base.convert(::Type{GammaShapeScale{T}}, dist::GammaDistributionsFamily) where {T}
    return GammaShapeScale(convert(T, shape(dist)), convert(T, scale(dist)))
end

function Base.convert(::Type{GammaShapeScale}, dist::GammaDistributionsFamily{T}) where {T}
    return convert(GammaShapeScale{T}, dist)
end

function Base.convert(::Type{GammaShapeRate{T}}, dist::GammaDistributionsFamily) where {T}
    return GammaShapeRate(convert(T, shape(dist)), convert(T, rate(dist)))
end

function Base.convert(::Type{GammaShapeRate}, dist::GammaDistributionsFamily{T}) where {T}
    return convert(GammaShapeRate{T}, dist)
end

BayesBase.default_prod_rule(::Type{<:GammaShapeRate}, ::Type{<:GammaShapeScale}) = PreserveTypeProd(Distribution)
BayesBase.default_prod_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeRate}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::GammaShapeRate, right::GammaShapeScale)
    return GammaShapeRate(shape(left) + shape(right) - 1, rate(left) + rate(right))
end

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::GammaShapeScale, right::GammaShapeRate)
    return GammaShapeScale(
        shape(left) + shape(right) - 1,
        (scale(left) * scale(right)) / (scale(left) + scale(right))
    )
end

function BayesBase.compute_logscale(
    new_dist::GammaDistributionsFamily,
    left_dist::GammaDistributionsFamily,
    right_dist::GammaDistributionsFamily
)
    ay, by = shape(new_dist), rate(new_dist)
    ax, bx = shape(left_dist), rate(left_dist)
    az, bz = shape(right_dist), rate(right_dist)
    return loggamma(ay) - loggamma(ax) - loggamma(az) + ax * log(bx) + az * log(bz) - ay * log(by)
end

# `E[log Γ(x)]` has no closed form. Stirling's formula writes `log Γ(x) = (x - 1/2) log x - x + log(2π)/2 + μ(x)`,
# where `μ` is Binet's remainder. The leading terms have a closed-form expectation, and `E[μ(x)]` is computed by
# quadrature in `w = log(x / E[x])`. In `w`, the log-density `a (1 + w - eʷ) + log(a)/2 - log(2π)/2 - μ(a)` has no
# cancellation for large shapes `a`, and `μ(x) ~ -log(x)/2` near zero is smooth.
function BayesBase.mean(::typeof(loggamma), dist::GammaDistributionsFamily)
    a, θ = shape(dist), scale(dist)
    m = mean(dist)
    stirling = (log2π - mean(log, dist)) / 2 + m * (digamma(a + 1) + log(θ) - 1)
    logm = log(m)
    lognorm = (log(a) - log2π) / 2 - binet_remainder(a, log(a))
    s = quadrature_breakpoint(sqrt(inv(a) + inv(a)^2)) # the standard deviation of `w`, roughly
    T = typeof(s)
    remainder, _ = quadgk(T(-Inf), -40s, -6s, zero(T), 6s, 40s, T(Inf); rtol = eps(T)^(3 // 4)) do w
        logp = a * (w - expm1(w)) + lognorm
        # `exp(-Inf)` would multiply infinite partial derivatives by zero
        isfinite(logp) || return zero(logp)
        return exp(logp) * binet_remainder(m * exp(w), w + logm)
    end
    return stirling + remainder
end

# Binet's remainder `μ(x) = log Γ(x) - (x - 1/2) log x + x - log(2π)/2`, given `z = log(x)`
function binet_remainder(x, z)
    if x >= 10
        r = inv(x)
        r2 = r * r
        return r / 12 - r * r2 / 360 + r * r2^2 / 1260 - r * r2^3 / 1680 + r * r2^4 / 1188
    end
    return loggamma(x + 1) + x - ((2x + 1) * z + log2π) / 2
end

# Breakpoints only place the quadrature nodes, so they drop the dual parts of `ForwardDiff` numbers
quadrature_breakpoint(x::AbstractFloat) = x
quadrature_breakpoint(x::ForwardDiff.Dual) = quadrature_breakpoint(ForwardDiff.value(x))
quadrature_breakpoint(x::Real) = float(x)

function Base.:(==)(left::GammaDistributionsFamily, right::GammaDistributionsFamily)
    return params(DefaultParametersSpace(), left) == params(DefaultParametersSpace(), right)
end

function Base.isapprox(left::GammaDistributionsFamily, right::GammaDistributionsFamily; kwargs...)
    return all(isapprox.(params(DefaultParametersSpace(), left), params(DefaultParametersSpace(), right); kwargs...))
end

# Natural parametrization

# Assume a single exponential family type tag both for `GammaShapeRate` and `GammaShapeScale`
# Thus both convert to `ExponentialFamilyDistribution{Gamma}`
exponential_family_typetag(::GammaDistributionsFamily) = Gamma

BayesBase.params(::DefaultParametersSpace, dist::GammaDistributionsFamily) = (shape(dist), scale(dist))

isproper(::DefaultParametersSpace, ::Type{Gamma}, θ, conditioner) = isnothing(conditioner) && (length(θ) === 2) && (all(>(0), θ))

function isproper(::NaturalParametersSpace, ::Type{Gamma}, η, conditioner)
    if length(η) !== 2
        return false
    end
    (η₁, η₂) = unpack_parameters(Gamma, η)
    return isnothing(conditioner) && (η₁ > -1) && (η₂ < 0)
end

function (::MeanToNatural{Gamma})(tuple_of_θ::Tuple{Any, Any})
    (shape, scale) = tuple_of_θ
    return (shape - 1, -inv(scale))
end

function (::NaturalToMean{Gamma})(tuple_of_η::Tuple{Any, Any})
    (η₁, η₂) = tuple_of_η
    return (η₁ + 1, -inv(η₂))
end

function unpack_parameters(::Type{Gamma}, packed)
    fi = firstindex(packed)
    si = firstindex(packed) + 1
    return (packed[fi], packed[si])
end

function BayesBase.mean(::typeof(inv), dist::GammaDistributionsFamily{T}) where {T}
    α = shape(dist)
    θ = rate(dist)
    return (α > 1 ? θ / (α - 1) : T(Inf))
end

isbasemeasureconstant(::Type{Gamma}) = ConstantBaseMeasure()

getbasemeasure(::Type{Gamma}) = (x) -> oneunit(x)
getsufficientstatistics(::Type{Gamma}) = (log, identity)

getlogpartition(::NaturalParametersSpace, ::Type{Gamma}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(Gamma, η)
    return loggamma(η₁ + one(η₁)) - (η₁ + one(η₁)) * log(-η₂)
end

getfisherinformation(::NaturalParametersSpace, ::Type{Gamma}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(Gamma, η)
    SA[trigamma(η₁ + one(η₁)) -inv(η₂); -inv(η₂) (η₁+one(η₁))/(η₂^2)]
end

getgradlogpartition(::NaturalParametersSpace, ::Type{Gamma}) = (η) -> begin
    (η₁, η₂) = unpack_parameters(Gamma, η)
    return SA[digamma(η₁+one(η₁))-log(-η₂), -(η₁+one(η₁))/η₂]
end

# Mean parametrization

getlogpartition(::DefaultParametersSpace, ::Type{Gamma}) = (θ) -> begin
    (shape, scale) = unpack_parameters(Gamma, θ)
    return loggamma(shape) + shape * log(scale)
end

getfisherinformation(::DefaultParametersSpace, ::Type{Gamma}) = (θ) -> begin
    (shape, scale) = unpack_parameters(Gamma, θ)
    return SA[
        trigamma(shape) inv(scale)
        inv(scale) shape/abs2(scale)
    ]
end

getgradlogpartition(::DefaultParametersSpace, ::Type{Gamma}) = (θ) -> begin
    (shape, scale) = unpack_parameters(Gamma, θ)
    return SA[digamma(shape)+log(scale), shape/scale]
end
