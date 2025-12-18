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

function Base.:(==)(left::GammaDistributionsFamily, right::GammaDistributionsFamily)
    return params(MeanParametersSpace(), left) == params(MeanParametersSpace(), right)
end

function Base.isapprox(left::GammaDistributionsFamily, right::GammaDistributionsFamily; kwargs...)
    return all(isapprox.(params(MeanParametersSpace(), left), params(MeanParametersSpace(), right); kwargs...))
end

# Natural parametrization

# Assume a single exponential family type tag both for `GammaShapeRate` and `GammaShapeScale`
# Thus both convert to `ExponentialFamilyDistribution{Gamma}`
exponential_family_typetag(::GammaDistributionsFamily) = Gamma

BayesBase.params(::MeanParametersSpace, dist::GammaDistributionsFamily) = (shape(dist), scale(dist))

isproper(::MeanParametersSpace, ::Type{Gamma}, θ, conditioner) = isnothing(conditioner) && (length(θ) === 2) && (all(>(0), θ))

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
    return SA[digamma(η₁ + one(η₁))-log(-η₂), -(η₁ + one(η₁))/η₂]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Gamma}) = (θ) -> begin
    (shape, scale) = unpack_parameters(Gamma, θ)
    return loggamma(shape) + shape * log(scale)
end

getfisherinformation(::MeanParametersSpace, ::Type{Gamma}) = (θ) -> begin
    (shape, scale) = unpack_parameters(Gamma, θ)
    return SA[
        trigamma(shape) inv(scale)
        inv(scale) shape/abs2(scale)
    ]
end

getgradlogpartition(::MeanParametersSpace, ::Type{Gamma}) = (θ) -> begin
    (shape, scale) = unpack_parameters(Gamma, θ)
    return SA[digamma(shape)-log(scale), -shape/scale]
end
