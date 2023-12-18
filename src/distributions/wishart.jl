export Wishart, WishartDistributionsFamily

import Distributions: Wishart
import Base: ndims, size, convert
import LinearAlgebra
import SpecialFunctions: digamma
import StatsFuns: logmvgamma
using LoopVectorization

"""
    WishartFast{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution

The `WishartFast` struct represents a fast version of the Wishart distribution. It is similar to the `Wishart` distribution from `Distributions.jl`, but it does not check input arguments, allowing the creation of improper `Wishart` messages.

For model creation and regular usage, it is recommended to use `Wishart` from `Distributions.jl`. The `WishartFast` distribution is intended for internal purposes and should not be directly used by regular users.

## Fields
- `ν::T`: The degrees of freedom parameter of the Wishart distribution.
- `invS::A`: The inverse scale matrix parameter of the Wishart distribution.

## Note

Internally, `WishartFast` stores and creates the inverse of its scale matrix. However, the `params()` function returns the scale matrix itself for backward compatibility. This is done to ensure better stability in the message passing update rules for `ReactiveMP.jl`.
"""
struct WishartFast{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν    :: T
    invS :: A
end

function WishartFast(ν::Real, invS::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(invS))
    return WishartFast(convert(T, ν), convert(AbstractArray{T}, invS))
end

WishartFast(ν::Integer, invS::AbstractMatrix{Real}) = WishartFast(float(ν), invS)

BayesBase.params(dist::WishartFast)  = (dist.ν, cholinv(dist.invS))
BayesBase.mean(dist::WishartFast)    = mean(convert(Wishart, dist))
BayesBase.var(dist::WishartFast)     = var(convert(Wishart, dist))
BayesBase.cov(dist::WishartFast)     = cov(convert(Wishart, dist))
BayesBase.std(dist::WishartFast)     = vmap(sqrt, var(dist))
BayesBase.mode(dist::WishartFast)    = mode(convert(Wishart, dist))
BayesBase.entropy(dist::WishartFast) = entropy(convert(Wishart, dist))

BayesBase.mean_cov(dist::WishartFast) = mean_cov(convert(Wishart, dist))

Base.size(dist::WishartFast)           = size(dist.invS)
Base.size(dist::WishartFast, dim::Int) = size(dist.invS, dim)

const WishartDistributionsFamily{T} = Union{Wishart{T}, WishartFast{T}}

function Base.convert(::Type{WishartFast{T}}, distribution::WishartFast) where {T}
    (ν, invS) = (distribution.ν, distribution.invS)
    return WishartFast(convert(T, ν), convert(AbstractMatrix{T}, invS))
end

function BayesBase.mean(::typeof(logdet), distribution::WishartFast)
    d       = size(distribution, 1)
    ν, invS = (distribution.ν, distribution.invS)
    T       = promote_type(typeof(ν), eltype(invS))
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(convert(T, 2)) - logdet(invS)
end

function BayesBase.mean(::typeof(logdet), distribution::Wishart)
    d    = size(distribution, 1)
    ν, S = params(distribution)
    T    = promote_type(typeof(ν), eltype(S))
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(convert(T, 2)) + logdet(S)
end

function BayesBase.mean(::typeof(inv), distribution::WishartDistributionsFamily)
    return mean(cholinv, distribution)
end

function BayesBase.mean(::typeof(cholinv), distribution::WishartFast)
    ν, invS = (distribution.ν, distribution.invS)
    return mean(InverseWishart(ν, invS))
end

function BayesBase.mean(::typeof(cholinv), distribution::Wishart)
    ν, S = params(distribution)
    return mean(InverseWishart(ν, cholinv(S)))
end

function BayesBase.convert_paramfloattype(::Type{T}, distribution::WishartFast) where {T}
    return WishartFast(convert_paramfloattype(T, distribution.ν), convert_paramfloattype(T, distribution.invS))
end

BayesBase.vague(::Type{<:Wishart}, dims::Int) = Wishart(dims, huge .* Array(Eye(dims)))

Base.ndims(dist::Wishart) = size(dist, 1)

function Base.convert(::Type{Wishart}, dist::WishartFast)
    (ν, S) = params(dist)
    return Wishart(ν, Matrix(Hermitian(S)))
end

function Base.convert(::Type{WishartFast}, dist::Wishart)
    (ν, S) = params(dist)
    return WishartFast(ν, cholinv(S))
end

function BayesBase.rand(rng::AbstractRNG, sampleable::WishartFast{T}) where {T}
    container = Matrix{Float64}(undef, size(sampleable))
    rand!(rng, sampleable, container)
end

function BayesBase.rand!(rng::AbstractRNG, sampleable::WishartFast{T}, x::AbstractMatrix) where {T}
    (df, S) = Distributions.params(sampleable)
    L = fastcholesky(S).L

    p = size(S, 1)
    singular = df <= p - 1
    if singular
        isinteger(df) || throw(ArgumentError("df of a singular Wishart distribution must be an integer (got $df)"))
    end

    A     = similar(S)
    l     = length(S)
    axes2 = axes(A, 2)
    r     = rank(S)

    if singular
        randn!(rng, view(A, :, view(axes2, 1:r)))
        fill!(view(A, :, view(axes2, (r+1):lastindex(axes2))), zero(eltype(A)))
    else
        Distributions._wishart_genA!(rng, A, df)
    end
    # Distributions.unwhiten!(S, A)
    lmul!(L, A)

    mul!(x, A, A', 1, 0)

    return x
end

function BayesBase.rand(rng::AbstractRNG, sampleable::WishartFast{T}, n::Int) where {T}
    container = [Matrix{T}(undef, size(sampleable)) for _ in 1:n]
    return rand!(rng, sampleable, container)
end

function BayesBase.rand!(rng::AbstractRNG, sampleable::WishartFast, x::AbstractVector{<:AbstractMatrix})
    # This is an adapted version of sampling from Distributions.jl
    (df, S) = Distributions.params(sampleable)
    L = fastcholesky(S).L

    p = size(S, 1)
    singular = df <= p - 1
    if singular
        isinteger(df) || throw(ArgumentError("df of a singular Wishart distribution must be an integer (got $df)"))
    end

    A     = similar(S)
    l     = length(S)
    axes2 = axes(A, 2)
    r     = rank(S)

    for C in x
        if singular
            randn!(rng, view(A, :, view(axes2, 1:r)))
            fill!(view(A, :, view(axes2, (r+1):lastindex(axes2))), zero(eltype(A)))
        else
            Distributions._wishart_genA!(rng, A, df)
        end
        # Distributions.unwhiten!(S, A)
        lmul!(L, A)

        mul!(C, A, A', 1, 0)
    end

    return x
end

function BayesBase.logpdf_sampling_optimized(dist::Wishart)
    optimized_dist = convert(WishartFast, dist)
    return (optimized_dist, optimized_dist)
end

function Distributions._logpdf(d::WishartFast, X::AbstractMatrix{<:Real})
    dist = convert(Wishart, d)
    return Distributions.logkernel(dist, X) + dist.logc0
end

# We do not define prod between `Wishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `WishartFast` messages as they are significantly faster in creation
BayesBase.params(::MeanParametersSpace, dist::WishartFast) = (dist.ν, dist.invS)
BayesBase.default_prod_rule(::Type{<:WishartFast}, ::Type{<:WishartFast}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::WishartFast, right::WishartFast)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two Wishart distributions of different sizes"

    d = size(left, 1)

    ldf, linvS = (left.ν, left.invS)
    rdf, rinvS = (right.ν, right.invS)

    # See Matrix Cookbook 
    # 3.2.5 The Searle Set of Identities - eq (163)
    # V  = lS * cholinv(lS + rS) * rS
    invV = linvS + rinvS
    df   = ldf + rdf - d - 1

    return WishartFast(df, invV)
end

function BayesBase.insupport(ef::ExponentialFamilyDistribution{WishartFast}, x::Matrix)
    return size(getindex(unpack_parameters(ef), 2)) == size(x) && isposdef(x)
end

# Natural parametrization

function isproper(::NaturalParametersSpace, ::Type{WishartFast}, η, conditioner)
    if !isnothing(conditioner) || length(η) <= 4 || any(isnan, η) || any(isinf, η)
        return false
    end

    (η1, η2) = unpack_parameters(WishartFast, η)
    # return  η1 > 0 && isposdef(-η2)
    return η1 > 0
end
function isproper(::MeanParametersSpace, ::Type{WishartFast}, θ, conditioner)
    if !isnothing(conditioner) || length(θ) <= 4 || any(isnan, θ) || any(isinf, θ)
        return false
    end

    (θ1, θ2) = unpack_parameters(WishartFast, θ)

    return θ1 > size(θ2, 1) - one(θ1)
    # return  θ1 > size(θ2,1) - one(θ1) && isposdef(θ2)
end

function (::MeanToNatural{WishartFast})(tuple_of_θ::Tuple{Any, Any})
    (ν, invS) = tuple_of_θ
    return ((ν - size(invS, 2) - one(ν)) / 2, -invS / 2)
end

function (::NaturalToMean{WishartFast})(tuple_of_η::Tuple{Any, Any})
    (η1, η2) = tuple_of_η
    return (2 * η1 + first(size(η2)) + 1, -2 * η2)
end

function unpack_parameters(::Type{WishartFast}, packed)
    len = length(packed)
    n = Int64(isqrt(len - 1))
    @inbounds η1 = packed[1]
    @inbounds η2 = reshape(view(packed, 2:len), n, n)

    return (η1, η2)
end

isbasemeasureconstant(::Type{WishartFast}) = ConstantBaseMeasure()

getbasemeasure(::Type{WishartFast}) = (x) -> one(Float64)
getsufficientstatistics(::Type{WishartFast}) = (chollogdet, identity)

getlogpartition(::NaturalParametersSpace, ::Type{WishartFast}) = (η) -> begin
    η1, η2 = unpack_parameters(WishartFast, η)
    p = first(size(η2))
    term1 = -(η1 + (p + one(η1)) / 2) * (logdet(-η2))
    term2 = logmvgamma(p, η1 + (p + one(η1)) / 2)
    return term1 + term2
end

mvdigamma(η,p) = sum( digamma(η + (one(d) - d)/2) for d=1:p)

getgradlogpartition(::NaturalParametersSpace, ::Type{WishartFast}) = (η) -> begin
    η1, η2 = unpack_parameters(WishartFast, η)
    p = first(size(η2))
    term1 = -logdet(-η2) + mvdigamma(η1 + (p + one(η1)) /2 , p)
    term2 = vec(((η1+(p+one(p))/2))*cholinv(η2))

    return [term1; term2]
end

getfisherinformation(::NaturalParametersSpace, ::Type{WishartFast}) =
    (η) -> begin
        η1, η2 = unpack_parameters(WishartFast, η)
        p = first(size(η2))
        invη2 = cholinv(η2)
        vinvη2 = -view(invη2, :)
        fimatrix = Matrix{Float64}(undef, p^2 + 1, p^2 + 1)
        @inbounds fimatrix[1, 1] = mvtrigamma(p, (η1 + (p + one(η1)) / 2))
        @inbounds fimatrix[1, 2:end] = vinvη2
        @inbounds fimatrix[2:end, 1] = vinvη2
        @inbounds fimatrix[2:end, 2:end] = (η1 + (p + one(η1)) / 2) * kron(invη2, invη2)
        return fimatrix
    end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{WishartFast}) = (θ) -> begin
    (ν, invS) = unpack_parameters(WishartFast, θ)
    p = first(size(invS))
    return (ν / 2) * (p * log(2.0) - logdet(invS)) + logmvgamma(p, ν / 2)
end

getgradlogpartition(::MeanParametersSpace, ::Type{WishartFast}) = (θ) -> begin
    ν, invS = unpack_parameters(WishartFast, θ)
    p = first(size(invS))
    term1 = ((p * log(2.0) - logdet(invS)) + mvdigamma(ν/2,p))/2
    term2 = vec((-ν/2)*cholinv(invS))

    return [term1; term2]
end

getfisherinformation(::MeanParametersSpace, ::Type{WishartFast}) = (θ) -> begin
    (df, invS) = unpack_parameters(WishartFast, θ)
    S = cholinv(invS)
    p = first(size(S))
    vinvS = 1 / 2 * vec(S)'
    fimatrix = Matrix{Float64}(undef, p^2 + 1, p^2 + 1)

    @inbounds fimatrix[1, 1] = mvtrigamma(p, df / 2) / 4
    @inbounds fimatrix[1, 2:end] = vinvS
    @inbounds fimatrix[2:end, 1] = vinvS
    @inbounds fimatrix[2:end, 2:end] = (df / 2) * kron(S, S)

    return fimatrix
end
