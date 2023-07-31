export Wishart

import Distributions: Wishart
import Base: ndims, size, convert
import LinearAlgebra
import SpecialFunctions: digamma

"""
    WishartFast

Same as `Wishart` from `Distributions.jl`, but does not check input arguments and allows creating improper `Wishart` message.
For model creation use `Wishart` from `Distributions.jl`. Regular user should never interact with `WishartFast`.

Note that internally `WishartFast` stores (and creates with) inverse of its-scale matrix, but (for backward compatibility) `params()` function returns the scale matrix itself. 
This is done for better stability in the message passing update rules for `ReactiveMP.jl`.
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

Distributions.params(dist::WishartFast)  = (dist.ν, cholinv(dist.invS))
Distributions.mean(dist::WishartFast)    = mean(convert(Wishart, dist))
Distributions.var(dist::WishartFast)     = var(convert(Wishart, dist))
Distributions.cov(dist::WishartFast)     = cov(convert(Wishart, dist))
Distributions.mode(dist::WishartFast)    = mode(convert(Wishart, dist))
Distributions.entropy(dist::WishartFast) = entropy(convert(Wishart, dist))

mean_cov(dist::WishartFast) = mean_cov(convert(Wishart, dist))

Base.size(dist::WishartFast)           = size(dist.invS)
Base.size(dist::WishartFast, dim::Int) = size(dist.invS, dim)

const WishartDistributionsFamily{T} = Union{Wishart{T}, WishartFast{T}}

to_marginal(dist::WishartFast) = convert(Wishart, dist)

function Base.convert(::Type{WishartFast{T}}, distribution::WishartFast) where {T}
    (ν, invS) = (distribution.ν, distribution.invS)
    return WishartFast(convert(T, ν), convert(AbstractMatrix{T}, invS))
end

function Distributions.mean(::typeof(logdet), distribution::WishartFast)
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

function Distributions.mean(::typeof(cholinv), distribution::WishartFast)
    ν, invS = (distribution.ν, distribution.invS)
    return mean(InverseWishart(ν, invS))
end

function Distributions.mean(::typeof(cholinv), distribution::Wishart)
    ν, S = params(distribution)
    return mean(InverseWishart(ν, cholinv(S)))
end

vague(::Type{<:Wishart}, dims::Int) = Wishart(dims, huge .* diageye(dims))

Base.ndims(dist::Wishart) = size(dist, 1)

function Base.convert(::Type{Wishart}, dist::WishartFast)
    (ν, S) = params(dist)
    return Wishart(ν, Matrix(Hermitian(S)))
end

function Base.convert(::Type{WishartFast}, dist::Wishart)
    (ν, S) = params(dist)
    return WishartFast(ν, cholinv(S))
end

function Distributions.rand(rng::AbstractRNG, sampleable::WishartFast{T}, n::Int) where {T}
    container = [Matrix{T}(undef, size(sampleable)) for _ in 1:n]
    return rand!(rng, sampleable, container)
end

function Distributions.rand!(rng::AbstractRNG, sampleable::WishartFast, x::AbstractVector{<:AbstractMatrix})
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
    optimized_dist = convert(WishartFast, dist)
    return (optimized_dist, optimized_dist)
end

# We do not define prod between `Wishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `WishartFast` messages as they are significantly faster in creation
closed_prod_rule(::Type{<:WishartFast}, ::Type{<:WishartFast}) = ClosedProd()

function Base.prod(::ClosedProd, left::WishartFast, right::WishartFast)
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

function pack_naturalparameters(dist::WishartFast)
    dof = dist.ν
    invscale = dist.invS
    p = first(size(invscale))

    return vcat((dof - p - 1) / 2, vec(-invscale / 2))
end

function pack_naturalparameters(dist::Wishart)
    dof = dist.df
    invscale = cholinv(dist.S)
    p = first(size(invscale))

    return vcat((dof - p - 1) / 2, vec(-invscale / 2))
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:WishartFast})
    η = getnaturalparameters(ef)
    len = length(η)
    n = Int64(isqrt(len-1))
    @inbounds η1 = η[1]
    @inbounds η2 = reshape(view(η,2:len),n,n)

    return η1, η2
end

check_valid_natural(::Type{<:Union{WishartFast, Wishart}}, params) = length(params) >= 5

Base.convert(::Type{ExponentialFamilyDistribution}, dist::WishartFast) = ExponentialFamilyDistribution(WishartFast, pack_naturalparameters(dist))

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Wishart) = ExponentialFamilyDistribution(WishartFast,pack_naturalparameters(dist) )
   
function Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{<:WishartFast})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    return WishartFast(2 * η1 + p + 1, -2η2)
end

function logpartition(ef::ExponentialFamilyDistribution{<:WishartFast})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    term1 = -(η1 + (p + 1) / 2) * logdet(-η2)
    term2 = logmvgamma(p, η1 + (p + 1) / 2)
    return term1 + term2
end

function isproper(ef::ExponentialFamilyDistribution{<:WishartFast})
    η1, η2 = unpack_naturalparameters(ef)
    isposdef(-η2) && (0 < η1)
end

mvtrigamma(p, x) = sum(trigamma(x + (1 - i) / 2) for i in 1:p)

function fisherinformation(dist::Wishart)
    df, S = dist.df, dist.S
    p = first(size(S))
    invS = inv(S)
    return [mvtrigamma(p, df / 2)/4 1/2*as_vec(invS)'; 1/2*as_vec(invS) df/2*kron(invS, invS)]
end

function fisherinformation(dist::WishartFast)
    df, invS = dist.ν, dist.invS
    p = first(size(invS))
    return [mvtrigamma(p, df / 2)/4 1/2*vec(invS)'; 1/2*vec(invS) df/2*kron(invS, invS)]
end

function fisherinformation(ef::ExponentialFamilyDistribution{<:WishartFast})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    invη2 = inv(η2)
    return [mvtrigamma(p, (η1 + (p + 1) / 2)) -vec(invη2)'; -vec(invη2) (η1+(p+1)/2)*kron(invη2, invη2)]
end

function insupport(ef::ExponentialFamilyDistribution{WishartFast, P, C, Safe}, x::Matrix) where {P, C}
    return size(getindex(unpack_naturalparameters(ef), 2)) == size(x) && isposdef(x)
end

function insupport(dist::WishartFast, x::Matrix)
    return (size(dist.invS) == size(x)) && isposdef(x)
end

basemeasure(::ExponentialFamilyDistribution{<:WishartFast}) = one(Float64)
function basemeasure(
    ::Union{<:ExponentialFamilyDistribution{<:WishartFast}, <:Union{WishartFast, Wishart}},
    x::Matrix
)
    return one(eltype(x))
end

sufficientstatistics(ef::ExponentialFamilyDistribution{<:WishartFast}) = (x) -> sufficientstatistics(ef,x)

function sufficientstatistics(
    ::Union{<:ExponentialFamilyDistribution{<:WishartFast}, <:Union{WishartFast, Wishart}},
    x::Matrix
)
    return vcat(chollogdet(x), vec(x))
end
