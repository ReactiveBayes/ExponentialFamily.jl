export Categorical
export logpartition

import Distributions: Categorical, probs
import LogExpFunctions: logsumexp
import FillArrays: OneElement

BayesBase.vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)
BayesBase.convert_paramfloattype(::Type{T}, distribution::Categorical) where {T <: Real} = Categorical(convert(AbstractVector{T}, probs(distribution)))

BayesBase.default_prod_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::Categorical, right::Categorical)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    mvec ./= sum(mvec)
    return Categorical(mvec; check_args = false)
end

BayesBase.probvec(dist::Categorical) = probs(dist)

function BayesBase.compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Categorical)
    return log(dot(probvec(left_dist), probvec(right_dist)))
end

# Natural parametrization

# The default implementation via `@generated` function fails to infer this
exponential_family_typetag(::Categorical) = Categorical

isproper(::NaturalParametersSpace, ::Type{Categorical}, η, conditioner) =
    isinteger(conditioner) && (conditioner === length(η)) && (length(η) >= 2) && (η[end] ≈ 0)
isproper(::MeanParametersSpace, ::Type{Categorical}, θ, conditioner) =
    isinteger(conditioner) && (conditioner === length(θ)) && (length(θ) >= 2) && all(>(0), θ) && isapprox(sum(θ), 1)

function separate_conditioner(::Type{Categorical}, params)
    # For `Categorical` we assume that the length of the vector it the `conditioner`
    # This is needed for sufficientstatistics statistics for example, it needs to know the length based on the type
    (p,) = params
    return (params, length(p))
end

function join_conditioner(::Type{Categorical}, cparams, conditioner)
    # For `Categorical` we assume that the length of the vector it the `conditioner`
    # But we don't really need to store it in the parameters
    return cparams
end

function (::MeanToNatural{Categorical})(tuple_of_θ::Tuple{Any}, _)
    (p,) = tuple_of_θ
    pₖ = p[end]
    return (map(pᵢ -> log(pᵢ / pₖ), p),)
end

function (::NaturalToMean{Categorical})(tuple_of_η::Tuple{V}, _) where {V <: Vector}
    (η,) = tuple_of_η
    return (softmax(η),)
end

# We use `Categorical` from `Distributions.jl` for the `MeanParametersSpace` 
# and their implementation supports only `Vector`s
function (::NaturalToMean{Categorical})(tuple_of_η::Tuple{V}, _) where {V <: AbstractVector}
    (η,) = tuple_of_η
    return (softmax(convert(Vector, η)),)
end

function unpack_parameters(::Type{Categorical}, packed, _)
    return unpack_parameters(Categorical, packed)
end

function unpack_parameters(::Type{Categorical}, packed)
    return (packed,)
end

getsupport(ef::ExponentialFamilyDistribution{Categorical}) = ClosedInterval{Int}(1, getconditioner(ef))

isbasemeasureconstant(::Type{Categorical}) = ConstantBaseMeasure()

getbasemeasure(::Type{Categorical}, _) = (x) -> oneunit(x)
getlogbasemeasure(::Type{Categorical}, _) = (x) -> zero(x)
getsufficientstatistics(::Type{Categorical}, conditioner) = ((x) -> OneElement(x, conditioner),)

getlogpartition(::NaturalParametersSpace, ::Type{Categorical}, conditioner) =
    (η) -> begin
        if (conditioner !== length(η))
            throw(
                DimensionMismatch(
                    lazy"Cannot evaluate the logparition of the `Categorical` with `conditioner = $(conditioner)` and vector of natural parameters `η = $(η)`"
                )
            )
        end
        return logsumexp(η)
    end

getgradlogpartition(::NaturalParametersSpace, ::Type{Categorical}, conditioner) =
    (η) -> begin
        if (conditioner !== length(η))
            throw(
                DimensionMismatch(
                    lazy"Cannot evaluate the logparition of the `Categorical` with `conditioner = $(conditioner)` and vector of natural parameters `η = $(η)`"
                )
            )
        end
        sumη = mapreduce(exp, +, η)
        return map(d -> exp(d) / sumη, η)
    end

getfisherinformation(::NaturalParametersSpace, ::Type{Categorical}, conditioner) =
    (η) -> begin
        if (conditioner !== length(η))
            throw(
                DimensionMismatch(
                    lazy"Cannot evaluate the fisherinformation matrix of the `Categorical` with `conditioner = $(conditioner)` and vector of natural parameters `η = $(η)`"
                )
            )
        end
        I = Matrix{eltype(η)}(undef, length(η), length(η))
        ∑expη = sum(exp, η)
        ∑expη² = abs2(∑expη)
        @inbounds for i in 1:length(η)
            expηᵢ = exp(η[i])
            I[i, i] = expηᵢ * (∑expη - expηᵢ) / ∑expη²
            for j in 1:(i-1)
                offv = -expηᵢ * exp(η[j]) / ∑expη²
                I[i, j] = offv
                I[j, i] = offv
            end
        end
        return I
    end

# Mean parametrization

# TODO: This function is AD unfriendly and gives wrong gradients and hessians
getlogpartition(::MeanParametersSpace, ::Type{Categorical}, conditioner) =
    (θ) -> begin
        if (conditioner !== length(θ))
            throw(
                DimensionMismatch(
                    lazy"Cannot evaluate the logparition of the `Categorical` with `conditioner = $(conditioner)` and vector of mean parameters `θ = $(θ)`"
                )
            )
        end
        return -log(θ[end])
    end

getfisherinformation(::MeanParametersSpace, ::Type{Categorical}, conditioner) =
    (θ) -> begin
        if (conditioner !== length(θ))
            throw(
                DimensionMismatch(
                    lazy"Cannot evaluate the fisherinformation matrix of the `Categorical` with `conditioner = $(conditioner)` and vector of mean parameters `θ = $(θ)`"
                )
            )
        end
        return Diagonal(map(inv, θ))
    end
