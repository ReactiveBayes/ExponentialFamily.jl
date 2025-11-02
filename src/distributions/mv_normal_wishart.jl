export MvNormalWishart
using Distributions
import StatsFuns: logmvgamma
using Random

"""
    MvNormalWishart{T, M <: AbstractArray{T}, V <: AbstractMatrix{T}, K <: Real, N <: Real} <: ContinuousMatrixDistribution

A multivariate normal-Wishart distribution, where `T` is the element type of the arrays `M` and matrices `V`, and `K` and `N` are real numbers. This distribution is a joint distribution of a multivariate normal random variable with mean `μ` and a Wishart-distributed random matrix with scale matrix `Ψ`, degrees of freedom `ν`, and the scalar `κ` as a scaling parameter.

# Fields
- `μ::M`: The mean vector of the multivariate normal distribution.
- `Ψ::V`: The scale matrix of the Wishart distribution.
- `κ::K`: The scaling parameter of the Wishart distribution.
- `ν::N`: The degrees of freedom of the Wishart distribution
"""
struct MvNormalWishart{T, M <: AbstractArray{T}, V <: AbstractMatrix{T}, K <: Real, N <: Real} <:
       ContinuousMultivariateMatrixvariateDistribution
    μ::M
    Ψ::V
    κ::K
    ν::N

    function MvNormalWishart(
        μ::M,
        Ψ::V,
        κ::K,
        ν::N
    ) where {T, M <: AbstractArray{T}, V <: AbstractMatrix{T}, K <: Real, N <: Real}
        new{T, M, V, K, N}(μ, Ψ, κ, ν)
    end
end

scatter(d::MvNormalWishart) = getindex(params(d), 2)
invscatter(d::MvNormalWishart) = cholinv(getindex(params(d), 2))
locationdim(d::MvNormalWishart) = length(d.μ)

BayesBase.dof(d::MvNormalWishart) = getindex(params(d), 4)
BayesBase.location(d::MvNormalWishart) = first(params(d))
BayesBase.params(d::MvNormalWishart) = (d.μ, d.Ψ, d.κ, d.ν)
BayesBase.scale(d::MvNormalWishart) = getindex(params(d), 3)
BayesBase.mean(d::MvNormalWishart) = (d.μ, d.ν * d.Ψ)
BayesBase.cov(d::MvNormalWishart) = var(d)

function BayesBase.std(d::MvNormalWishart)
    c = d.ν - locationdim(d) + one(d.ν)
    ψ = d.Ψ * (d.κ + one(d.κ)) / (d.κ * c)
    return (real(sqrt((c / (c - 2)) * ψ)), real(sqrt(cov(Wishart(d.ν, d.Ψ)))))
end

function BayesBase.var(d::MvNormalWishart)
    c = d.ν - locationdim(d) + one(d.ν)
    ψ = d.Ψ * (d.κ + one(d.κ)) / (d.κ * c)
    return ((c / (c - 2)) * ψ, cov(Wishart(d.ν, d.Ψ)))
end

function locationdim(ef::ExponentialFamilyDistribution{MvNormalWishart})
    len = length(getnaturalparameters(ef))
    return Int64((-1 + isqrt(1 - 4 * (2 - len))) / 2)
end

function BayesBase.pdf(dist::MvNormalWishart, x::Tuple)
    μ, Ψ, κ, ν = params(dist)
    x1, x2 = x
    return pdf(MvNormalMeanPrecision(μ, κ * x2), x1) * pdf(Wishart(ν, Ψ), x2)
end

BayesBase.logpdf(dist::MvNormalWishart, x::Tuple) = log(pdf(dist, x))

function BayesBase.rand!(rng::AbstractRNG, dist::MvNormalWishart, container::Tuple{AbstractVector, AbstractMatrix})
    μ, Ψ, κ, ν = params(dist)
    rand!(rng, Wishart(ν, Ψ), container[2])
    rand!(rng, MvNormalMeanPrecision(μ, κ * container[2]), container[1])
    return container
end

function BayesBase.rand!(rng::AbstractRNG, dist::MvNormalWishart, container::AbstractVector{T}) where {T <: Tuple}
    for i in eachindex(container)
        rand!(rng, dist, container[i])
    end
    return container
end

function BayesBase.rand(rng::AbstractRNG, dist::MvNormalWishart{T}) where {T}
    container = (Vector{T}(undef, locationdim(dist)), Matrix{T}(undef, (locationdim(dist), locationdim(dist))))
    return rand!(rng, dist, container)
end

function BayesBase.rand(rng::AbstractRNG, dist::MvNormalWishart{T}, nsamples::Int64) where {T}
    container = Vector{Tuple{Vector{T}, Matrix{T}}}(undef, nsamples)
    for i in eachindex(container)
        container[i] = (Vector{T}(undef, locationdim(dist)), Matrix{T}(undef, (locationdim(dist), locationdim(dist))))
        rand!(rng, dist, container[i])
    end
    return container
end

BayesBase.default_prod_rule(::Type{<:MvNormalWishart}, ::Type{<:MvNormalWishart}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalWishart, right::MvNormalWishart)
    μleft, Sleft, λleft, νleft = params(left)
    μright, Sright, λright, νright = params(right)

    λ = λleft + λright
    ν = νleft + νright - locationdim(left)

    μhat = λleft * μleft + λright * μright
    μ = μhat / λ

    Sinvleft = cholinv(Sleft)
    Sinvright = cholinv(Sright)

    S = cholinv(Sinvleft + λleft * μleft * μleft' + Sinvright + λright * μright * μright' - μhat * μhat' / λ)

    return MvNormalWishart(μ, S, λ, ν)
end

function BayesBase.insupport(::ExponentialFamilyDistribution{MvNormalWishart}, x)
    return isposdef(getindex(x, 2))
end

function isproper(::NaturalParametersSpace, ::Type{MvNormalWishart}, η, conditioner)
    if !isnothing(conditioner) || length(η) < 8 || any(isnan, η) || any(isinf, η)
        return false
    end

    (η1, η2, η3, η4) = unpack_parameters(MvNormalWishart, η)
    # return  η1 > 0 && isposdef(-η2)
    return η3 < 0 && η4 > -1 / 2
end

function isproper(::DefaultParametersSpace, ::Type{MvNormalWishart}, θ, conditioner)
    if !isnothing(conditioner) || length(θ) <= 8 || any(isnan, θ) || any(isinf, θ)
        return false
    end

    (θ1, θ2, θ3, θ4) = unpack_parameters(MvNormalWishart, θ)

    return θ4 > length(θ1) - one(θ4) && θ3 > 0
    # return  θ1 > size(θ2,1) - one(θ1) && isposdef(θ2)
end

function (::MeanToNatural{MvNormalWishart})(tuple_of_θ::Tuple{Any, Any, Any, Any})
    (μ, S, λ, ν) = tuple_of_θ
    return (λ * μ, (-1 / 2) * (cholinv(S) + λ * μ * μ'), -λ / 2, (ν - length(μ)) / 2)
end

function (::NaturalToMean{MvNormalWishart})(tuple_of_η::Tuple{Any, Any, Any, Any})
    (η1, η2, η3, η4) = tuple_of_η

    return (-(1 / 2) * η1 / η3, cholinv(-2 * η2 + (1 / 2) * η1 * η1' / (η3)), -2 * η3, length(η1) + 2 * η4)
end

function unpack_parameters(::Type{MvNormalWishart}, η)
    len = length(η)
    d = Int64((-1 + isqrt(1 - 4 * (2 - len))) / 2)

    @inbounds η1 = view(η, 1:d)
    @inbounds η2 = reshape(view(η, (d+1):(d^2+d)), d, d)
    @inbounds η3 = η[d^2+d+1]
    @inbounds η4 = η[d^2+d+2]

    return η1, η2, η3, η4
end

isbasemeasureconstant(::Type{MvNormalWishart}) = ConstantBaseMeasure()

getbasemeasure(::Type{MvNormalWishart}) = (x) -> 1.0

function getsufficientstatistics(::Type{MvNormalWishart})
    return (
        (z) -> begin
            (x, S) = z
            return S * x
        end, (z) -> begin
            (_, S) = z
            return S
        end, (z) -> begin
            (x, S) = z
            return dot3arg(x, S, x)
        end, (z) -> begin
            (_, S) = z
            return logdet(S)
        end
    )
end

getlogpartition(::NaturalParametersSpace, ::Type{MvNormalWishart}) = (η) -> begin
    η1, η2, η3, η4 = unpack_parameters(MvNormalWishart, η)
    d = length(η1)
    term1 = -(d * (1 / 2)) * log(-2 * (η3))
    term2 = -((d + 2 * η4) * (1 / 2)) * logdet(-2 * η2 + (1 / 2) * η1 * η1' / (η3))
    term3 = log(2.0) * d * (d + 2 * η4) * (1 / 2)
    term4 = logmvgamma(d, (d + 2 * η4) * (1 / 2))

    return (term1 + term2 + term3 + term4) + (d / 2)log2π
end

getgradlogpartition(::NaturalParametersSpace, ::Type{MvNormalWishart}) =
    (η) -> begin
        η1, η2, η3, η4 = unpack_parameters(MvNormalWishart, η)
        d = length(η1)
        const1 = -(d + 2η4) / 2
        kronecker = kron(η1, η1')
        veckronecker = vec(kronecker)

        const2 = cholinv(-2η2 + kronecker / (2η3))
        vconst2 = vec(const2)
        kronright = kron(Eye(d), η1) / (2η3)
        kronleft = kron(η1, Eye(d)) / (2η3)
        dη2 = -2 * const1 * const2
        dη1 = const1 * (kronright + kronleft)' * vconst2
        dη3 = (-d / (2η3)) - const1 * dot(vconst2, veckronecker / (2η3^2))
        dη4 = -logdet(-2η2 + kronecker / (2η3)) + d * log(2) + mvdigamma((d + 2 * η4) * (1 / 2), d)

        return vcat(dη1, vec(dη2), dη3, dη4)
    end

getfisherinformation(::NaturalParametersSpace, ::Type{MvNormalWishart}) =
    (η) -> begin
        η1, η2, η3, η4 = unpack_parameters(MvNormalWishart, η)
        d = length(η1)

        ϕ = -2η2 + kron(η1, η1') / (2η3)
        invϕ = cholinv(ϕ)
        kroninv = kron(invϕ, invϕ)
        kronright = kron(Eye(d), η1) / (2η3)
        kronleft = kron(η1, Eye(d)) / (2η3)
        vinvϕ = vec(cholinv(ϕ))
        constant = -(d + 2η4) / 2
        kronη1 = vec(kron(η1, η1'))

        fimatrix = zeros(d^2 + d + 2, d^2 + d + 2)

        @inbounds begin
            ##diagonals
            fimatrix[1:d, 1:d] = -constant * ((kroninv * (kronright + kronleft))' * (kronleft + kronright)) + constant * invϕ / η3
            fimatrix[(d+1):(d^2+d), (d+1):(d^2+d)] = -4 * constant * kroninv
            fimatrix[d^2+d+1, d^2+d+1] = constant * vinvϕ' * (kronη1) / η3^3 - (constant / (4 * η3^4)) * (kroninv * kronη1)' * kronη1 + d / (2η3^2)
            fimatrix[d^2+d+2, d^2+d+2] = mvtrigamma(d, -constant)

            ##offdiagonals
            fimatrix[1:d, (d+1):(d^2+d)] = 2constant * (kroninv * (kronright + kronleft))'
            fimatrix[(d+1):(d^2+d), 1:d] = fimatrix[1:d, (d+1):(d^2+d)]'
            fimatrix[1:d, d^2+d+1] = (constant) * ((kroninv * (kronleft + kronright))' * kronη1 / (2 * η3^2) - (kronleft + kronright)' * vinvϕ / η3)
            fimatrix[d^2+d+1, 1:d] = fimatrix[1:d, d^2+d+1]'
            fimatrix[1:d, end] = -vinvϕ' * (kronleft + kronright)
            fimatrix[end, 1:d] = fimatrix[1:d, end]'
            fimatrix[(d+1):(d^2+d), d^2+d+1] = -constant / (η3^2) * kroninv * kronη1
            fimatrix[d^2+d+1, (d+1):(d^2+d)] = fimatrix[(d+1):(d^2+d), d^2+d+1]'
            fimatrix[(d+1):(d^2+d), d^2+d+2] = 2vinvϕ
            fimatrix[d^2+d+2, (d+1):(d^2+d)] = fimatrix[(d+1):(d^2+d), d^2+d+2]'
            fimatrix[d^2+d+1, d^2+d+2] = vinvϕ' * kronη1 / (2η3^2)
            fimatrix[d^2+d+2, d^2+d+1] = fimatrix[d^2+d+1, d^2+d+2]
        end
        return fimatrix
    end

# Mean parametrization

getlogpartition(::DefaultParametersSpace, ::Type{MvNormalWishart}) = (θ) -> begin
    (μ, S, λ, ν) = unpack_parameters(MvNormalWishart, θ)
    d = length(μ)

    term1 = -(d * (1 / 2)) * log(λ)
    term2 = (ν * (1 / 2)) * logdet(S)
    term3 = log(2.0) * d * ν / 2
    term4 = logmvgamma(d, ν * (1 / 2))

    return term1 + term2 + term3 + term4 + (d / 2)log2π
end

getfisherinformation(::DefaultParametersSpace, ::Type{MvNormalWishart}) = (θ) -> begin
    μ, T, κ, ν = unpack_parameters(MvNormalWishart, θ)
    d = length(μ)

    kronT = kron(T, T)
    fimatrix = zeros(d^2 + d + 2, d^2 + d + 2)

    @inbounds begin
        ##diagonals
        fimatrix[1:d, 1:d] = ν * κ * T
        fimatrix[(d+1):(d^2+d), (d+1):(d^2+d)] = (ν / 2) * kronT
        fimatrix[d^2+d+1, d^2+d+1] = d / (2κ^2)
        fimatrix[d^2+d+2, d^2+d+2] = mvtrigamma(d, ν / 2) / 4

        # fimatrix[d+1:d^2+d, d^2+d+2] = -vec(inv(T))/2
        # fimatrix[d^2+d+2, d+1:d^2+d] = fimatrix[d+1:d^2+d, d^2+d+2]'
    end

    return fimatrix

    # return blockdiag(sparse(ν*κ*T) , sparse((ν/2)*kronT) , sparse(Diagonal([d/(2κ^2), mvtrigamma(d, ν/2)/4])))
end

function _logpdf(ef::ExponentialFamilyDistribution{MvNormalWishart}, x)
    # TODO: Think of what to do with this assert
    @assert insupport(ef, x)
    _logpartition = logpartition(ef)
    return _logpdf(ef, x, _logpartition)
end

function _logpdf(ef::ExponentialFamilyDistribution{MvNormalWishart}, x, logpartition)
    # TODO: Think of what to do with this assert
    @assert insupport(ef, x)
    η = getnaturalparameters(ef)
    # Use `_` to avoid name collisions with the actual functions
    _statistics = sufficientstatistics(ef, x)
    _basemeasure = basemeasure(ef, x)
    return log(_basemeasure) + dot(η, flatten_parameters(MvNormalWishart, _statistics)) - logpartition
end

function _pdf(ef::ExponentialFamilyDistribution{MvNormalWishart}, x)
    exp(_logpdf(ef, x))
end
