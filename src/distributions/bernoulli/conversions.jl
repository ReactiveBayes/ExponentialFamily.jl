
function Base.convert(::Type{Distribution}, exponentialfamily::ExponentialFamilyDistribution{Bernoulli})
    (logprobability,) = unpack_naturalparameters(exponentialfamily)
    return Bernoulli(logistic(logprobability))
end

function Base.convert(::Type{ExponentialFamilyDistribution}, dist::Bernoulli)
    @assert !(succprob(dist) ≈ 1) "Bernoulli natural parameters are not defiend for p = 1."
    return ExponentialFamilyDistribution(Bernoulli, pack_naturalparameters(dist))
end