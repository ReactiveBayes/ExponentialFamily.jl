export Laplace
using Distributions
import Distributions: Laplace, params, logpdf
using DomainSets
using StaticArrays

vague(::Type{<:Laplace}) = Laplace(0.0, huge)

# The default product between two `Laplace` objects is `PreserveTypeProd(Laplace)`,
# which is possible only if the location parameters match
default_prod_rule(::Type{<:Laplace}, ::Type{<:Laplace}) = PreserveTypeProd(Laplace)

function Base.prod(::PreserveTypeProd{Laplace}, left::Laplace, right::Laplace)
    location_left, scale_left = params(left)
    location_right, scale_right = params(right)

    if isapprox(location_left, location_right)
        return Laplace(location_left, scale_left * scale_right / (scale_left + scale_right))
    end

    error("""
        Cannot compute a closed product of two `Laplace` distribution with different location parameters.
        To compute a generic product in the natural parameters space, convert both distributions to the 
        `ExponentialFamilyDistribution` type and use the `PreserveTypeProd(ExponentialFamilyDistribution)`
        prod strategy.
    """)
end

# The default product between two `ExponentialFamilyDistribution{Laplace}` objects is 
# `ProdPreserveType(ExponentialFamilyDistribution{Laplace})`, which is possible only if the location parameters match
default_prod_rule(::Type{<:ExponentialFamilyDistribution{T}}, ::Type{<:ExponentialFamilyDistribution{T}}) where {T <: Laplace} =
    PreserveTypeProd(ExponentialFamilyDistribution{Laplace})

function Base.prod!(container::ExponentialFamilyDistribution{Laplace}, left::ExponentialFamilyDistribution{Laplace}, right::ExponentialFamilyDistribution{Laplace})
    (η_container, conditioner_container) = (getnaturalparameters(container), getconditioner(container))
    (η_left, conditioner_left) = (getnaturalparameters(left), getconditioner(left))
    (η_right, conditioner_right) = (getnaturalparameters(right), getconditioner(right))

    if isapprox(conditioner_left, conditioner_right) && isapprox(conditioner_left, conditioner_container)
        LoopVectorization.vmap!(+, η_container, η_left, η_right)
        return container
    end

    error("""
        Cannot compute a closed product of two `Laplace` distribution in their natural parametrization with different conditioners (location parameter).
        To compute a generic product in the natural parameters space, convert both distributions to the 
        `ExponentialFamilyDistribution` type and use the `PreserveTypeProd(ExponentialFamilyDistribution)`
        prod strategy.
    """)
end

function Base.prod(::PreserveTypeProd{ExponentialFamilyDistribution{Laplace}}, left::ExponentialFamilyDistribution{Laplace}, right::ExponentialFamilyDistribution{Laplace})
    return prod!(similar(left), left, right)
end

# function Base.prod(
#     ::PreserveTypeProd{ExponentialFamilyDistribution},
#     ef_left::ExponentialFamilyDistribution{T},
#     ef_right::ExponentialFamilyDistribution{T}
# ) where {T <: Laplace}
#     (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
#     (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
#     if conditioner_left == conditioner_right
#         return ExponentialFamilyDistribution(Laplace, η_left + η_right, conditioner_left)
#     else
#         basemeasure = (x) -> one(x)
#         sufficientstatistics = (x) -> SA[abs(x - conditioner_left), abs(x - conditioner_right)]
#         sorted_conditioner = sort(SA[conditioner_left, conditioner_right])
#         function logpartition(η)
#             A1 = exp(η[1] * conditioner_left + η[2] * conditioner_right)
#             A2 = exp(-η[1] * conditioner_left + η[2] * conditioner_right)
#             A3 = exp(-η[1] * conditioner_left - η[2] * conditioner_right)
#             B1 = (exp(sorted_conditioner[2] * (-η[1] - η[2])) - 1.0) / (-η[1] - η[2])
#             B2 =
#                 (exp(sorted_conditioner[1] * (η[1] - η[2])) - exp(sorted_conditioner[2] * (η[1] - η[2]))) /
#                 (η[1] - η[2])
#             B3 = (1.0 - exp(sorted_conditioner[1] * (η[1] + η[2]))) / (η[1] + η[2])

#             return log(A1 * B1 + A2 * B2 + A3 * B3)
#         end
#         naturalparameters = vcat(η_left, η_right)
#         supp = RealInterval{Float64}(-Inf, Inf)

#         return ExponentialFamilyDistribution(
#             Univariate,
#             naturalparameters,
#             nothing,
#             basemeasure,
#             sufficientstatistics,
#             logpartition,
#             supp
#         )
#     end
# end

# Natural parametrization

function isproper(::NaturalParametersSpace, ::Type{Laplace}, η, conditioner::Number) 
    if isnan(conditioner) || isinf(conditioner) || length(η) !== 1
        return false
    end

    (η₁, ) = unpack_parameters(Laplace, η)

    return !isnan(η₁) && !isinf(η₁) && η₁ < 0 
end

function isproper(::MeanParametersSpace, ::Type{Laplace}, θ, conditioner::Number) 
    if isnan(conditioner) || isinf(conditioner) || length(θ) !== 1
        return false
    end

    (scale, ) = unpack_parameters(Laplace, θ)

    return !isnan(scale) && !isinf(scale) && scale > 0 
end

function separate_conditioner(::Type{Laplace}, params)
    location, scale = params
    return ((scale, ), location)
end

function join_conditioner(::Type{Laplace}, cparams, conditioner) 
    (scale, ) = cparams
    location = conditioner
    return (location, scale)
end

function (::MeanToNatural{Laplace})(tuple_of_θ::Tuple{Any}, _)
    (scale,) = tuple_of_θ
    return (-inv(scale), )
end

function (::NaturalToMean{Laplace})(tuple_of_η::Tuple{Any}, _)
    (η₁,) = tuple_of_η
    return (-inv(η₁), )
end

function unpack_parameters(::Type{Laplace}, packed) 
    return (first(packed), )
end

isbasemeasureconstant(::Type{Laplace}) = ConstantBaseMeasure()

getbasemeasure(::Type{Laplace}, _) = (x) -> oneunit(x)
getsufficientstatistics(::Type{Laplace}, conditioner) = (
    (x) -> abs(x - conditioner),
)

getlogpartition(::NaturalParametersSpace, ::Type{Laplace}, _) = (η) -> begin 
    (η₁,) = unpack_parameters(Laplace, η)
    return log(-2 / η₁)
end

getfisherinformation(::NaturalParametersSpace, ::Type{Laplace}, _) = (η) -> begin 
    (η₁,) = unpack_parameters(Laplace, η)
    return SA[ inv(η₁^2);; ]
end

# Mean parametrization

getlogpartition(::MeanParametersSpace, ::Type{Laplace}, _) = (θ) -> begin 
    (scale, ) = unpack_parameters(Laplace, θ)
    return log(2scale)
end

getfisherinformation(::MeanParametersSpace, ::Type{Laplace}, _) = (θ) -> begin 
    (scale,) = unpack_parameters(Laplace, θ)
    return SA[ inv(abs2(scale));; ] # 1 / scale^2
end
