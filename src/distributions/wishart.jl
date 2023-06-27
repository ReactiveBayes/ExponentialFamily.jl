export Wishart

import Distributions: Wishart
import Base: ndims, size, convert
import LinearAlgebra
import SpecialFunctions: digamma

"""
    WishartImproper

Same as `Wishart` from `Distributions.jl`, but does not check input arguments and allows creating improper `Wishart` message.
For model creation use `Wishart` from `Distributions.jl`. Regular user should never interact with `WishartImproper`.

Note that internally `WishartImproper` stores (and creates with) inverse of its-scale matrix, but (for backward compatibility) `params()` function returns the scale matrix itself. 
This is done for better stability in the message passing update rules for `ReactiveMP.jl`.
"""
struct WishartImproper{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν    :: T
    invS :: A
end

function WishartImproper(ν::Real, invS::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(invS))
    return WishartImproper(convert(T, ν), convert(AbstractArray{T}, invS))
end

WishartImproper(ν::Integer, invS::AbstractMatrix{Real}) = WishartImproper(float(ν), invS)

Distributions.params(dist::WishartImproper)  = (dist.ν, cholinv(dist.invS))
Distributions.mean(dist::WishartImproper)    = mean(convert(Wishart, dist))
Distributions.var(dist::WishartImproper)     = var(convert(Wishart, dist))
Distributions.cov(dist::WishartImproper)     = cov(convert(Wishart, dist))
Distributions.mode(dist::WishartImproper)    = mode(convert(Wishart, dist))
Distributions.entropy(dist::WishartImproper) = entropy(convert(Wishart, dist))

mean_cov(dist::WishartImproper) = mean_cov(convert(Wishart, dist))

Base.size(dist::WishartImproper)           = size(dist.invS)
Base.size(dist::WishartImproper, dim::Int) = size(dist.invS, dim)

const WishartDistributionsFamily{T} = Union{Wishart{T}, WishartImproper{T}}

to_marginal(dist::WishartImproper) = convert(Wishart, dist)

function Base.convert(::Type{WishartImproper{T}}, distribution::WishartImproper) where {T}
    (ν, invS) = (distribution.ν, distribution.invS)
    return WishartImproper(convert(T, ν), convert(AbstractMatrix{T}, invS))
end

function Distributions.mean(::typeof(logdet), distribution::WishartImproper)
    d       = size(distribution, 1)
    ν, invS = (distribution.ν, distribution.invS)
    T       = promote_type(typeof(ν), eltype(invS))
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(convert(T, 2)) - logdet(invS)
end

function Distributions.mean(::typeof(logdet), distribution::Wishart)
    d    = size(distribution, 1)
    ν, S = params(distribution)
    T    = promote_type(typeof(ν), eltype(S))
    return mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(convert(T, 2)) + logdet(S)
end

function Distributions.mean(::typeof(inv), distribution::WishartDistributionsFamily)
    return mean(cholinv, distribution)
end

function Distributions.mean(::typeof(cholinv), distribution::WishartImproper)
    ν, invS = (distribution.ν, distribution.invS)
    return mean(InverseWishart(ν, invS))
end

function Distributions.mean(::typeof(cholinv), distribution::Wishart)
    ν, S = params(distribution)
    return mean(InverseWishart(ν, cholinv(S)))
end

vague(::Type{<:Wishart}, dims::Int) = Wishart(dims, huge .* diageye(dims))

Base.ndims(dist::Wishart) = size(dist, 1)

function Base.convert(::Type{Wishart}, dist::WishartImproper)
    (ν, S) = params(dist)
    return Wishart(ν, cholinv(Matrix(Hermitian(S))))
end

function Base.convert(::Type{WishartImproper}, dist::Wishart)
    (ν, S) = params(dist)
    return WishartImproper(ν, cholinv(S))
end

function Distributions.rand(rng::AbstractRNG, sampleable::WishartImproper{T}, n::Int) where {T}
    container = [Matrix{T}(undef, size(sampleable)) for _ in 1:n]
    return rand!(rng, sampleable, container)
end

function Distributions.rand!(rng::AbstractRNG, sampleable::WishartImproper, x::AbstractVector{<:AbstractMatrix})
    # This is an adapted version of sampling from Distributions.jl
    (df, S) = Distributions.params(sampleable)
    L = Distributions.PDMats.chol_lower(fastcholesky(S))

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

function logpdf_sample_optimized(dist::Wishart)
    optimized_dist = convert(WishartImproper, dist)
    return (optimized_dist, optimized_dist)
end

# We do not define prod between `Wishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `WishartImproper` messages as they are significantly faster in creation
prod_closed_rule(::Type{<:WishartImproper}, ::Type{<:WishartImproper}) = ClosedProd()

function Base.prod(::ClosedProd, left::WishartImproper, right::WishartImproper)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two Wishart distributions of different sizes"

    d = size(left, 1)

    ldf, linvS = (left.ν, left.invS)
    rdf, rinvS = (right.ν, right.invS)

    # See Matrix Cookbook 
    # 3.2.5 The Searle Set of Identities - eq (163)
    # V  = lS * cholinv(lS + rS) * rS
    invV = linvS + rinvS
    df   = ldf + rdf - d - 1

    return WishartImproper(df, invV)
end

check_valid_natural(::Type{<:Union{WishartImproper, Wishart}}, params) = length(params) === 2

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::WishartImproper)
    dof = dist.ν
    invscale = dist.invS
    p = first(size(invscale))
    return KnownExponentialFamilyDistribution(WishartImproper, [(dof - p - 1) / 2, -invscale / 2])
end

function Base.convert(::Type{KnownExponentialFamilyDistribution}, dist::Wishart)
    dof = dist.df
    invscale = cholinv(dist.S)
    p = first(size(invscale))
    return KnownExponentialFamilyDistribution(WishartImproper, [(dof - p - 1) / 2, -invscale / 2])
end

function Base.convert(::Type{Distribution}, params::KnownExponentialFamilyDistribution{<:WishartImproper})
    η = getnaturalparameters(params)
    η1 = first(η)
    η2 = getindex(η, 2)
    p = first(size(η2))

    if isproper(params)
        return Wishart(2 * η1 + p + 1, 0.5cholinv(-η2))
    end
    return WishartImproper(2 * η1 + p + 1, 0.5cholinv(-η2))
end

function logpartition(params::KnownExponentialFamilyDistribution{<:WishartImproper})
    η = getnaturalparameters(params)
    η1 = first(η)
    η2 = getindex(η, 2)
    p = first(size(η2))
    term1 = -(η1 + (p + 1) / 2) * logdet(-η2)
    term2 = logmvgamma(p, η1 + (p + 1) / 2)
    return term1 + term2
end

function isproper(params::KnownExponentialFamilyDistribution{<:WishartImproper})
    η = getnaturalparameters(params)
    η1 = first(η)
    η2 = getindex(η, 2)
    isposdef(-η2) && (0 < η1)
end

mvtrigamma(p, x) = sum(trigamma(x + (1 - i) / 2) for i in 1:p)

function fisherinformation(dist::Wishart)
    df, S = dist.df, dist.S
    p = first(size(S))
    invS = inv(S)
    return [mvtrigamma(p, df / 2)/4 1/2*as_vec(invS)'; 1/2*as_vec(invS) df/2*kron(invS, invS)]
end

function fisherinformation(params::KnownExponentialFamilyDistribution{<:WishartImproper})
    η = getnaturalparameters(params)
    η1 = first(η)
    η2 = getindex(η, 2)
    p = first(size(η2))
    invη2 = inv(η2)
    return [mvtrigamma(p, (η1 + (p + 1) / 2)) -as_vec(invη2)'; -as_vec(invη2) (η1+(p+1)/2)*kron(invη2, invη2)]
end

function insupport(ef::KnownExponentialFamilyDistribution{WishartImproper, P, C, Safe}, x::Matrix) where {P, C}
    return size(getindex(getnaturalparameters(ef), 2)) == size(x) && isposdef(x)
end

function insupport(dist::WishartImproper, x::Matrix)
    return (size(dist.invS) == size(x)) && isposdef(x)
end

function basemeasure(
    union::Union{<:KnownExponentialFamilyDistribution{<:WishartImproper}, <:Union{WishartImproper, Wishart}},
    x::Matrix
)
    @assert insupport(union, x) "$(x) is not in the support of Wishart"
    return 1.0
end
function sufficientstatistics(
    union::Union{<:KnownExponentialFamilyDistribution{<:WishartImproper}, <:Union{WishartImproper, Wishart}},
    x::Matrix
)
    @assert insupport(union, x) "$(x) is not in the support of Wishart"
    return [chollogdet(x), x]
end
