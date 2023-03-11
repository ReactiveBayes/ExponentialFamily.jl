using Distributions

struct NaturalParameters{T,P}
    params::P
    NaturalParameters(::Type{T}, params::P) where {T,P}  = begin 
        @assert check_valid_natural(T, params) == true "Parameter vector $(params) is not a valid natural parameter for distribution $(T)"
        new{T,P}(params)
    end
end

function check_valid_natural end

get_params(np::NaturalParameters) = np.params

Base.convert(::Type{T}, params::NaturalParameters) where {T <: Distribution} = Base.convert(T, Base.convert(Distribution, params))

Base.:+(left::NaturalParameters{T}, right::NaturalParameters{T}) where {T} = NaturalParameters(T, get_params(left) + get_params(right))
Base.:-(left::NaturalParameters{T}, right::NaturalParameters{T}) where {T} = NaturalParameters(T, get_params(left) - get_params(right))
Base.:(==)(left::NaturalParameters{T}, right::NaturalParameters{T}) where {T} = get_params(left) == get_params(right) 

Distributions.logpdf(np::NaturalParameters{T}, x) where {T} = Distributions.logpdf(Base.convert(T, np), x)
Distributions.pdf(np::NaturalParameters{T}, x) where {T} = Distributions.pdf(Base.convert(T, np), x)
Distributions.cdf(np::NaturalParameters{T}, x) where {T} = Distributions.cdf(Base.convert(T, np), x)

