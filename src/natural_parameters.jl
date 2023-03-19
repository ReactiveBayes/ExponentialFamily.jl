using Distributions

struct ExponentialFamilyDistribution{T, P, C}
    naturalparameters::P
    conditioner::C
    ExponentialFamilyDistribution(::Type{T}, naturalparameters::P, conditioner::C = nothing) where {T, P, C} = begin
        @assert check_valid_natural(T, naturalparameters) == true "Parameter vector $(naturalparameters) is not a valid natural parameter for distribution $(T)"
        @assert check_valid_conditioner(T, conditioner) "$(conditioner) is not a valid conditioner for distribution $(T) or 'check_valid_conditioner' function is not implemented!"
        new{T, P, C}(naturalparameters, conditioner)
    end
end

check_valid_conditioner(::Type{T}, conditioner) where {T} = conditioner === nothing

function check_valid_natural end

getnaturalparameters(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.naturalparameters
getconditioner(exponentialfamily::ExponentialFamilyDistribution) = exponentialfamily.conditioner

Base.convert(::Type{T}, naturalparameters::ExponentialFamilyDistribution) where {T <: Distribution} =
    Base.convert(T, Base.convert(Distribution, naturalparameters))

Base.:+(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution) = +(plus(left, right), left, right)
Base.:-(left::ExponentialFamilyDistribution, right::ExponentialFamilyDistribution) = -(plus(left, right), left, right)

Base.:+(::Plus, left::ExponentialFamilyDistribution{T1}, right::ExponentialFamilyDistribution{T2}) where {T1, T2} =
    ExponentialFamilyDistribution(T1, getnaturalparameters(left) + getnaturalparameters(right), getconditioner(left))
Base.:-(::Plus, left::ExponentialFamilyDistribution{T1}, right::ExponentialFamilyDistribution{T2}) where {T1, T2} =
    ExponentialFamilyDistribution(T1, getnaturalparameters(left) - getnaturalparameters(right), getconditioner(left))

Base.:+(::Concat, left::ExponentialFamilyDistribution{T1}, right::ExponentialFamilyDistribution{T2}) where {T1, T2} = [left, right]
Base.:-(::Concat, left::ExponentialFamilyDistribution{T1}, right::ExponentialFamilyDistribution{T2}) where {T1, T2} =
    [left, ExponentialFamilyDistribution(T2, -getnaturalparameters(right), getconditioner(right))]

Base.:(==)(left::ExponentialFamilyDistribution{T}, right::ExponentialFamilyDistribution{T}) where {T} =
    getnaturalparameters(left) == getnaturalparameters(right) && getconditioner(left) == getconditioner(right)

function Base.:(≈)(left::ExponentialFamilyDistribution{T1}, right::ExponentialFamilyDistribution{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return ≈(
        ExponentialFamilyDistribution(T, getnaturalparameters(left), getconditioner(left)),
        ExponentialFamilyDistribution(T, getnaturalparameters(right), getconditioner(right))
    )
end

Base.:(≈)(left::ExponentialFamilyDistribution{T}, right::ExponentialFamilyDistribution{T}) where {T} =
    getnaturalparameters(left) ≈ getnaturalparameters(right) && getconditioner(left) == getconditioner(right)

Distributions.logpdf(exponentialfamily::ExponentialFamilyDistribution, x) = Distributions.logpdf(Base.convert(Distribution, exponentialfamily), x)
Distributions.pdf(exponentialfamily::ExponentialFamilyDistribution, x) = Distributions.pdf(Base.convert(Distribution, exponentialfamily), x)
Distributions.cdf(exponentialfamily::ExponentialFamilyDistribution, x) = Distributions.cdf(Base.convert(Distribution, exponentialfamily), x)

"""
Everywhere in the package, we stick to a convention that we represent exponential family distributions in the following form:

``f_X(x\\mid\\theta) = h(x)\\,\\exp\\!\\bigl[\\,\\eta(\\theta) \\cdot T(x) - A(\\theta)\\,\\bigr]``.

So the `lognormalizer` sign should align with this form.
"""
function lognormalizer end

Base.prod(::ProdAnalytical, left::ExponentialFamilyDistribution{T1}, right::ExponentialFamilyDistribution{T2}) where {T1,T2} = Base.prod(plus(left,right),left,right)

function Base.prod(::Plus,left::ExponentialFamilyDistribution{T1}, right::ExponentialFamilDistribution{T2}) where {T1, T2} 
    ef_left = Base.convert(ExponentialFamilyDistribution, left)
    ef_right = Base.convert(ExponentialFamilyDistribution, right)
    naturalparams = getnaturalparameters(ef_left) + getnaturalparameters(ef_right)
    return Base.convert(Distribution, ExponentialFamilyDistribution(T1,naturalparams,getconditioner(left)))
end