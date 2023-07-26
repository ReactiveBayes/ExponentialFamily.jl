export InverseWishart

import Distributions: InverseWishart, Wishart, pdf!
import Base: ndims, size, convert
import LinearAlgebra
import StatsFuns: logπ, logmvgamma
import SpecialFunctions: digamma, loggamma

"""
    InverseWishartImproper

Same as `InverseWishart` from `Distributions.jl`, but does not check input arguments and allows creating improper `InverseWishart` message.
For model creation use `InverseWishart` from `Distributions.jl`. Regular user should never interact with `InverseWishartImproper`.
"""
struct InverseWishartImproper{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν::T
    S::A
end

function InverseWishartImproper(ν::Real, S::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(S))
    return InverseWishartImproper(convert(T, ν), convert(AbstractArray{T}, S))
end

InverseWishartImproper(ν::Integer, S::AbstractMatrix{Real}) = InverseWishartImproper(float(ν), S)

Distributions.params(dist::InverseWishartImproper) = (dist.ν, dist.S)
Distributions.mean(dist::InverseWishartImproper)   = mean(convert(InverseWishart, dist))
Distributions.var(dist::InverseWishartImproper)    = var(convert(InverseWishart, dist))
Distributions.cov(dist::InverseWishartImproper)    = cov(convert(InverseWishart, dist))
Distributions.mode(dist::InverseWishartImproper)   = mode(convert(InverseWishart, dist))

mean_cov(dist::InverseWishartImproper) = mean_cov(convert(InverseWishart, dist))

Base.size(dist::InverseWishartImproper)           = size(dist.S)
Base.size(dist::InverseWishartImproper, dim::Int) = size(dist.S, dim)

const InverseWishartDistributionsFamily{T} = Union{InverseWishart{T}, InverseWishartImproper{T}}

to_marginal(dist::InverseWishartImproper) = convert(InverseWishart, dist)

function Base.convert(::Type{InverseWishartImproper{T}}, distribution::InverseWishartImproper) where {T}
    (ν, S) = params(distribution)
    return InverseWishartImproper(convert(T, ν), convert(AbstractMatrix{T}, S))
end

# from "Parametric Bayesian Estimation of Differential Entropy and Relative Entropy" Gupta et al.
function Distributions.entropy(dist::InverseWishartImproper)
    d = size(dist, 1)
    ν, S = params(dist)
    d * (d - 1) / 4 * logπ + mapreduce(i -> loggamma((ν + 1.0 - i) / 2), +, 1:d) + ν / 2 * d +
    (d + 1) / 2 * (logdet(S) - log(2)) -
    (ν + d + 1) / 2 * mapreduce(i -> digamma((ν - d + i) / 2), +, 1:d)
end

function Distributions.mean(::typeof(logdet), dist::InverseWishartImproper)
    d = size(dist, 1)
    ν, S = params(dist)
    return -(mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) - logdet(S))
end

function Distributions.mean(::typeof(inv), dist::InverseWishartImproper)
    return mean(cholinv, dist)
end

function Distributions.mean(::typeof(cholinv), dist::InverseWishartImproper)
    ν, S = params(dist)
    return mean(Wishart(ν, cholinv(S)))
end

function Distributions.rand(rng::AbstractRNG, sampleable::InverseWishartImproper{T}, n::Int) where {T}
    container = [Matrix{T}(undef, size(sampleable)) for _ in 1:n]
    return rand!(rng, sampleable, container)
end

function Distributions.rand!(rng::AbstractRNG, sampleable::InverseWishartImproper, x::AbstractVector{<:AbstractMatrix})
    # This is an adapted version of sampling from Distributions.jl
    (df, S⁻¹) = Distributions.params(sampleable)
    S = cholinv(S⁻¹)
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

        M = Cholesky(A, 'L', convert(LinearAlgebra.BlasInt, 0))
        LinearAlgebra.inv!(M)

        copyto!(C, 1, M.L.data, 1, l)
    end

    return x
end

function Distributions.pdf!(
    out::AbstractArray{<:Real},
    distribution::InverseWishartImproper,
    samples::AbstractArray{<:AbstractMatrix{<:Real}, O}
) where {O}
    @assert length(out) === length(samples) "Invalid dimensions in pdf!"

    p = size(distribution, 1)
    (df, Ψ) = Distributions.params(distribution)

    T = copy(Ψ)
    R = similar(T)
    l = length(T)

    M = fastcholesky!(T)

    h_df = df / 2
    Ψld = logdet(M)
    logc0 = -h_df * (p * convert(typeof(df), Distributions.logtwo) - Ψld) - logmvgamma(p, h_df)

    @inbounds for i in 1:length(out)
        copyto!(T, 1, samples[i], 1, l)
        C = fastcholesky!(T)
        ld = logdet(C)
        LinearAlgebra.inv!(C)
        mul!(R, Ψ, C.factors)
        r = tr(R)
        out[i] = exp(-0.5 * ((df + p + 1) * ld + r) + logc0)
    end

    return out
end

vague(::Type{<:InverseWishart}, dims::Integer) = InverseWishart(dims + 2, tiny .* diageye(dims))

Base.ndims(dist::InverseWishart) = size(dist, 1)

function Base.convert(::Type{InverseWishart}, dist::InverseWishartImproper)
    (ν, S) = params(dist)
    return InverseWishart(ν, Matrix(Hermitian(S)))
end

Base.convert(::Type{InverseWishartImproper}, dist::InverseWishart) = InverseWishartImproper(params(dist)...)

function logpdf_sample_optimized(dist::InverseWishart)
    optimized_dist = convert(InverseWishartImproper, dist)
    return (optimized_dist, optimized_dist)
end

# We do not define prod between `InverseWishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `InverseWishartImproper` messages as they are significantly faster in creation
closed_prod_rule(::Type{<:InverseWishartImproper}, ::Type{<:InverseWishartImproper}) = ClosedProd()

function Base.prod(::ClosedProd, left::InverseWishartImproper, right::InverseWishartImproper)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two InverseWishart distributions of different sizes"

    d = size(left, 1)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V = lS + rS

    df = ldf + rdf + d + 1

    return InverseWishartImproper(df, V)
end


function pack_naturalparameters(dist::Union{InverseWishartImproper, InverseWishart})
    dof, scale  = params(dist)
    p = first(size(scale))

    return vcat(-(dof + p + 1) / 2, vec(-scale / 2))
end

function unpack_naturalparameters(ef::ExponentialFamilyDistribution{<:InverseWishartImproper})
    η = getnaturalparameters(ef)
    len = length(η)
    n = Int64(isqrt(len-1))
    @inbounds η1 = η[1]
    @inbounds η2 = reshape(view(η,2:len),n,n)

    return η1, η2
end

check_valid_natural(::Type{<:Union{InverseWishartImproper, InverseWishart}}, params) = length(params) >= 5

Base.convert(::Type{ExponentialFamilyDistribution}, dist::Union{InverseWishartImproper, InverseWishart}) = 
    ExponentialFamilyDistribution(InverseWishartImproper, pack_naturalparameters(dist))

function Base.convert(::Type{Distribution}, ef::ExponentialFamilyDistribution{<:InverseWishartImproper})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    return InverseWishart(-(2 * η1 + p + 1), -2 * η2)
end

function logpartition(ef::ExponentialFamilyDistribution{<:InverseWishartImproper})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    term1 = (η1 + (p + 1) / 2) * logdet(-η2)
    term2 = logmvgamma(p, -(η1 + (p + 1) / 2))
    return term1 + term2
end

function isproper(ef::ExponentialFamilyDistribution{<:InverseWishartImproper})
    η1, η2 = unpack_naturalparameters(ef)
    isposdef(-η2) && (η1 < 0)
end

function fisherinformation(ef::ExponentialFamilyDistribution{<:InverseWishartImproper})
    η1, η2 = unpack_naturalparameters(ef)
    p = first(size(η2))
    invη2 = inv(η2)
    return [mvtrigamma(p, (η1 + (p + 1) / 2)) -vec(invη2)'; -vec(invη2) (η1+(p+1)/2)*kron(invη2, invη2)]
end

function fisherinformation(dist::InverseWishart)
    ν = dist.df
    S = dist.Ψ
    p = first(size(S))
    invscale = inv(S)

    hessian = ones(eltype(S), p^2 + 1, p^2 + 1)
    hessian[1, 1] = mvtrigamma(p, -ν / 2) / 4
    hessian[1, 2:p^2+1] = as_vec(invscale) / 2
    hessian[2:p^2+1, 1] = as_vec(invscale) / 2
    hessian[2:p^2+1, 2:p^2+1] = ν / 2 * kron(invscale, invscale)
    hessian[2:p^2+1, 2:p^2+1] = -1 * hessian[2:p^2+1, 2:p^2+1]
    return hessian
end

function insupport(ef::ExponentialFamilyDistribution{InverseWishartImproper, P, C, Safe}, x::Matrix) where {P, C}
    return size(getindex(unpack_naturalparameters(ef), 2)) == size(x) && isposdef(x)
end

function basemeasure(
    ef::ExponentialFamilyDistribution{<:InverseWishartImproper},
    x
)
    @assert insupport(ef, x) "$(x) is not in the support of inverse Wishart"
    return 1.0
end

function sufficientstatistics(
    ef::ExponentialFamilyDistribution{<:InverseWishartImproper},
    x
)
    @assert insupport(ef, x) "$(x) is not in the support of inverse Wishart"
    return vcat(chollogdet(x), vec(cholinv(x)))
end
