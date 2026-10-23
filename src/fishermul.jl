export fishermul, fishermul!, getfishermul, getfishermul!, FisherMulBuffer

"""
    fishermul([space = NaturalParametersSpace()], ef, [η], v)

Return the Fisher-information–vector product `F(η) * v` in packed parameter
coordinates. Omitted `η` defaults to `getnaturalparameters(ef)`. Only
[`NaturalParametersSpace`](@ref) is currently implemented. Scalable built-in
families apply the Fisher matrix without constructing it.

See also [`fishermul!`](@ref), [`getfishermul`](@ref), [`FisherMulBuffer`](@ref).
"""
fishermul(ef::ExponentialFamilyDistribution, args...) = fishermul(NaturalParametersSpace(), ef, args...)
fishermul(space, ef::ExponentialFamilyDistribution, v::AbstractVector) = fishermul(space, ef, getnaturalparameters(ef), v)
fishermul(space, ef::ExponentialFamilyDistribution, η::AbstractVector, v::AbstractVector) = getfishermul(space, ef)(η, v)

"""
    FisherMulBuffer([space = NaturalParametersSpace()], ef, [η], v)

Allocate reusable workspace for [`fishermul!`](@ref). The buffer is tied to the
parameter space, family, conditioner/dimensions, packed length, and promoted
element type of `η` and `v`. Parameter and direction values may change between
calls; all parameter-dependent workspace is refreshed. Do not share a buffer
between concurrent calls or access its internal storage.

With a compatible buffer, built-in families allocate no heap memory on warmed-up
calls with valid dense Float32/Float64 arrays. This guarantee excludes other numeric
types, error paths, and user-defined Fisher implementations. A buffer constructor
is not supplied for generic allocating Fisher-matrix fallbacks.

On Julia 1.10/1.11, `VonMises` and `VonMisesFisher` incur small fixed allocations
inside `SpecialFunctions.besseli` (64 and 32 bytes per product, respectively).
"""
struct FisherMulBuffer{Space, Family, R, C, B}
    conditioner::C
    length::Int
    storage::B
end

FisherMulBuffer(ef::ExponentialFamilyDistribution, args...) = FisherMulBuffer(NaturalParametersSpace(), ef, args...)
FisherMulBuffer(space, ef::ExponentialFamilyDistribution, v::AbstractVector) = FisherMulBuffer(space, ef, getnaturalparameters(ef), v)
function FisherMulBuffer(space::NaturalParametersSpace, ef::ExponentialFamilyDistribution{T}, η::AbstractVector, v::AbstractVector) where {T}
    isnothing(getattributes(ef)) || throw(ArgumentError("Buffered Fisher multiplication is not implemented for these custom attributes."))
    return _fishermul_buffer(space, T, getconditioner(ef), η, v)
end

"""
    fishermul!(w, [space = NaturalParametersSpace()], ef, [η], v, buffer = nothing)

Overwrite `w` with `F(η) * v` and return `w`. The inputs `η` and `v` are preserved;
`w` must not overlap either input. All vectors use the distribution's packed
parameter order. Length mismatches throw `DimensionMismatch`; overlapping
storage or an incompatible buffer family/type throws `ArgumentError`.

If `buffer` is `nothing`, temporary workspace is allocated internally when needed.
Pass a compatible [`FisherMulBuffer`](@ref) to reuse workspace. A generic fallback
uses the existing Fisher matrix and may allocate. Only natural space is currently
implemented; omitting `space` selects it.
"""
fishermul!(w::AbstractVector, ef::ExponentialFamilyDistribution, args...) = fishermul!(w, NaturalParametersSpace(), ef, args...)
function fishermul!(w::AbstractVector, space, ef::ExponentialFamilyDistribution, v::AbstractVector, buffer::Union{Nothing, FisherMulBuffer} = nothing)
    return fishermul!(w, space, ef, getnaturalparameters(ef), v, buffer)
end
function fishermul!(
    w::AbstractVector,
    space,
    ef::ExponentialFamilyDistribution,
    η::AbstractVector,
    v::AbstractVector,
    buffer::Union{Nothing, FisherMulBuffer} = nothing
)
    return getfishermul!(space, ef)(w, η, v, buffer)
end

"""
    getfishermul([space = NaturalParametersSpace()], ef)
    getfishermul([space = NaturalParametersSpace()], T, [conditioner])

Return a callable `(η, v) -> F(η) * v`. Dispatch follows [`getfisherinformation`](@ref),
including conditioners and shape metadata for matrix families. Only natural
space is currently implemented.
"""
getfishermul(ef::ExponentialFamilyDistribution) = getfishermul(NaturalParametersSpace(), ef)
getfishermul(::Type{T}, conditioner = nothing) where {T <: Distribution} = getfishermul(NaturalParametersSpace(), T, conditioner)
getfishermul(space::NaturalParametersSpace, ef::ExponentialFamilyDistribution) = _getfishermul(space, getattributes(ef), ef)
_getfishermul(space, ::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getfishermul(space, T, getconditioner(ef))
function _getfishermul(::NaturalParametersSpace, ::ExponentialFamilyDistributionAttributes, ef)
    F = getfisherinformation(ef)
    return (η, v) -> begin
        _check_fishermul_inputs(η, v)
        F(η) * v
    end
end
getfishermul(space::NaturalParametersSpace, ::Type{T}, ::Nothing) where {T <: Distribution} = getfishermul(space, T)
getfishermul(::NaturalParametersSpace, ::Type{T}) where {T <: Distribution} = _fishermul_getter(T, nothing)
getfishermul(::NaturalParametersSpace, ::Type{T}, conditioner) where {T <: Distribution} = _fishermul_getter(T, conditioner)
function _fishermul_getter(::Type{T}, conditioner) where {T}
    return (η, v) -> begin
        _check_fishermul_inputs(η, v)
        _fishermul_return(T, conditioner, η, v)
    end
end

"""
    getfishermul!([space = NaturalParametersSpace()], ef)
    getfishermul!([space = NaturalParametersSpace()], T, [conditioner])

Return a callable `(w, η, v, buffer = nothing)` with the contract of
[`fishermul!`](@ref). The callable does not own mutable workspace; callers may
provide a [`FisherMulBuffer`](@ref).
"""
getfishermul!(ef::ExponentialFamilyDistribution) = getfishermul!(NaturalParametersSpace(), ef)
getfishermul!(::Type{T}, conditioner = nothing) where {T <: Distribution} = getfishermul!(NaturalParametersSpace(), T, conditioner)
getfishermul!(space::NaturalParametersSpace, ef::ExponentialFamilyDistribution) = _getfishermul!(space, getattributes(ef), ef)
_getfishermul!(space, ::Nothing, ef::ExponentialFamilyDistribution{T}) where {T} = getfishermul!(space, T, getconditioner(ef))
function _getfishermul!(::NaturalParametersSpace, ::ExponentialFamilyDistributionAttributes, ef)
    F = getfisherinformation(ef)
    return (w, η, v, buffer = nothing) -> begin
        _check_fishermul_output(w, η, v)
        isnothing(buffer) || throw(ArgumentError("Buffered Fisher multiplication is not implemented for these custom attributes."))
        mul!(w, F(η), v)
        w
    end
end
getfishermul!(space::NaturalParametersSpace, ::Type{T}, ::Nothing) where {T <: Distribution} = getfishermul!(space, T)
getfishermul!(space::NaturalParametersSpace, ::Type{T}) where {T <: Distribution} = _fishermul_getter!(space, T, nothing)
getfishermul!(space::NaturalParametersSpace, ::Type{T}, conditioner) where {T <: Distribution} = _fishermul_getter!(space, T, conditioner)
function _fishermul_getter!(space::NaturalParametersSpace, ::Type{T}, conditioner) where {T}
    return (w, η, v, buffer = nothing) -> _fishermul_apply!(w, space, T, conditioner, η, v, buffer)
end

@inline function _check_fishermul_inputs(η, v)
    Base.require_one_based_indexing(η, v)
    length(η) == length(v) || throw(DimensionMismatch("The Fisher direction must have the same packed length as the parameters."))
    return nothing
end

@inline function _check_fishermul_output(w, η, v)
    _check_fishermul_inputs(η, v)
    Base.require_one_based_indexing(w)
    length(w) == length(v) || throw(DimensionMismatch("The Fisher output and direction must have equal lengths."))
    (Base.mightalias(w, η) || Base.mightalias(w, v)) && throw(ArgumentError("The Fisher output must not overlap the parameters or direction."))
    return nothing
end

_fishermul_eltype(η, v) = float(promote_type(eltype(η), eltype(v)))

# The generic path preserves StaticArrays multiplication for fixed-size families.
_fishermul_return(::Type{T}, c, η, v) where {T} = getfisherinformation(NaturalParametersSpace(), T, c)(η) * v
function _fishermul_return_direct(::Type{T}, c, η, v) where {T}
    w = similar(v, _fishermul_eltype(η, v))
    return _fishermul_apply!(w, NaturalParametersSpace(), T, c, η, v, nothing)
end

struct _FisherMulFallback end
_fishermul_storage(::Type, c, η, ::Type{R}) where {R} = _FisherMulFallback()

function _fishermul_buffer(space::NaturalParametersSpace, ::Type{T}, c, η, v) where {T}
    _check_fishermul_inputs(η, v)
    R = _fishermul_eltype(η, v)
    storage = _fishermul_storage(T, c, η, R)
    if storage isa _FisherMulFallback
        F = getfisherinformation(space, T, c)(η)
        F isa StaticMatrix || throw(ArgumentError("This family has no buffered Fisher-product implementation."))
    end
    return FisherMulBuffer{typeof(space), T, R, typeof(c), typeof(storage)}(c, length(η), storage)
end

@inline function _fishermul_apply!(w, space::NaturalParametersSpace, ::Type{T}, c, η, v, ::Nothing) where {T}
    _check_fishermul_output(w, η, v)
    storage = _fishermul_storage(T, c, η, _fishermul_eltype(η, v))
    _fishermul_kernel!(w, T, c, η, v, storage)
    return w
end

@inline function _fishermul_apply!(w, space::NaturalParametersSpace, ::Type{T}, c, η, v, buffer::FisherMulBuffer{S, F, R}) where {T, S, F, R}
    _check_fishermul_output(w, η, v)
    (S === typeof(space) && F === T && R === _fishermul_eltype(η, v)) ||
        throw(ArgumentError("The Fisher buffer has an incompatible space, family, or numeric type."))
    buffer.length == length(η) || throw(DimensionMismatch("The Fisher buffer has an incompatible packed length."))
    if !isequal(buffer.conditioner, c)
        c isa Tuple && throw(DimensionMismatch("The Fisher buffer has an incompatible shape."))
        throw(ArgumentError("The Fisher buffer has an incompatible conditioner."))
    end
    _fishermul_kernel!(w, T, c, η, v, buffer.storage)
    return w
end

_fishermul_kernel!(w, ::Type{T}, c, η, v, ::_FisherMulFallback) where {T} = mul!(w, getfisherinformation(NaturalParametersSpace(), T, c)(η), v)

# Copy into caller-owned storage before factoring: the parameter vector is read-only.
function _fishermul_inverse!(A)
    C = fastcholesky!(A)
    if eltype(A) <: LinearAlgebra.BlasFloat
        LinearAlgebra.inv!(C)
    else
        copyto!(A, inv(C))
    end
    return A
end

function _fishermul_inverse!(A, source, scale)
    @inbounds for i in eachindex(A, source)
        A[i] = scale * source[i]
    end
    return _fishermul_inverse!(A)
end

function _fishermul_bilinear(x, A, y)
    result = zero(promote_type(eltype(x), eltype(A), eltype(y)))
    @inbounds for j in eachindex(y), i in eachindex(x)
        result += x[i] * A[i, j] * y[j]
    end
    return result
end
