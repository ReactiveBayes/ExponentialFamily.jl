using Distributions

struct NaturalParameters{T, P, C}
    params::P
    conditioner::C
    NaturalParameters(::Type{T}, params::P, conditioner::C = nothing) where {T, P, C} = begin
        @assert check_valid_natural(T, params) == true "Parameter vector $(params) is not a valid natural parameter for distribution $(T)"
        new{T, P, C}(params, conditioner)
    end
end

function check_valid_natural end

get_params(np::NaturalParameters) = np.params
get_conditioner(np::NaturalParameters) = np.conditioner

Base.convert(::Type{T}, params::NaturalParameters) where {T <: Distribution} =
    Base.convert(T, Base.convert(Distribution, params))

Base.:+(left::NaturalParameters, right::NaturalParameters) = +(plus(left, right), left, right)
Base.:-(left::NaturalParameters, right::NaturalParameters) = -(plus(left, right), left, right)

Base.:+(::Plus, left::NaturalParameters{T1}, right::NaturalParameters{T2}) where{T1, T2} = NaturalParameters(T1, get_params(left) + get_params(right))
Base.:-(::Plus, left::NaturalParameters{T1}, right::NaturalParameters{T2}) where{T1, T2} = NaturalParameters(T1, get_params(left) - get_params(right))

Base.:+(::Concat, left::NaturalParameters{T1}, right::NaturalParameters{T2}) where{T1, T2} = [left, right]
Base.:-(::Concat, left::NaturalParameters{T1}, right::NaturalParameters{T2}) where{T1, T2} = [left, NaturalParameters(T2,-get_params(right))]

function Base.:+(::Conditioned, left::NaturalParameters{T1}, right::NaturalParameters{T2}) where{T1, T2} 
    @assert !isnothing(get_conditioner(left)) && !isnothing(get_conditioner(right)) "$(left) and $(right) should have specificed conditioner"
    @assert get_conditioner(left) == get_conditioner(right) "Conditioners must be the same to add the natural parameters of $(left) andd $(right)"
    return NaturalParameters(T1,get_params(left) + get_params(right),get_conditioner(left))
end
function Base.:-(::Conditioned, left::NaturalParameters{T1}, right::NaturalParameters{T2}) where{T1, T2} 
    @assert !isnothing(get_conditioner(left)) && !isnothing(get_conditioner(right)) "$(left) and $(right) should have specificed conditioner"
    @assert get_conditioner(left) == get_conditioner(right) "Conditioners must be the same to subtract the natural parameters of $(left) andd $(right)"
    return NaturalParameters(T1,get_params(left) - get_params(right),get_conditioner(left))
end

Base.:(==)(left::NaturalParameters{T}, right::NaturalParameters{T}) where {T} = get_params(left) == get_params(right)

function Base.:(≈)(left::NaturalParameters{T1}, right::NaturalParameters{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return ≈(NaturalParameters(T, get_params(left)), NaturalParameters(T, get_params(right)))
end

Base.:(≈)(left::NaturalParameters{T}, right::NaturalParameters{T}) where {T} = get_params(left) ≈ get_params(right)

Distributions.logpdf(np::NaturalParameters, x) = Distributions.logpdf(Base.convert(Distribution, np), x)
Distributions.pdf(np::NaturalParameters, x) = Distributions.pdf(Base.convert(Distribution, np), x)
Distributions.cdf(np::NaturalParameters, x) = Distributions.cdf(Base.convert(Distribution, np), x)
