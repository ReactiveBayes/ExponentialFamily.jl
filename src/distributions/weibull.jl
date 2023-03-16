import Distributions: Weibull

check_valid_natural(::Type{<:Weibull}, params) = length(params) === 1
check_valid_conditioner(::Type{<:Weibull}, conditioner) = isreal(conditioner) && conditioner > 0

function isproper(params::NaturalParameters{Weibull})
    η = get_params(params)
    return first(η) < 0
end

