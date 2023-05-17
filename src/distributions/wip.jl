using Distributions
using ExponentialFamily
using SpecialFunctions
using Zygote
import SpecialFunctions: log2π, besselix, besselk

import ExponentialFamily: KnownExponentialFamilyDistribution, getnaturalparameters, logpartition, basemeasure, fisherinformation
import SpecialFunctions: besselj

function pdf(dist::VonMisesFisher, x::Vector{T}) where {T<:Real}
    d = first(size(dist))
    μ = mean(dist)
    κ = concentration(dist)

    hp = T(d/2)
    q = hp - 1

    c = (κ^q) / ((2π)^hp * besselix(q, κ)*exp(κ))

    return c * exp(κ * (μ' * x))
end

function logconstant(κ, p)
    T = typeof(κ)
    hp = T(p/2)
    q = hp - 1
    q * log(κ) - hp * log2π - log(besselix(q, κ)) - κ
end

function constant(κ, p)
    T = typeof(κ)
    hp = T(p/2)
    q = hp - 1
    κ^q / (2π^hp* besselix(q, κ)*exp(κ))
end

dist = VonMisesFisher(ones(3))

pdf(dist, zeros(3))

exp(Distributions._logpdf(dist, zeros(3)))

a = randn(4)
a = a/norm(a)
ef = convert(KnownExponentialFamilyDistribution, VonMisesFisher(a, 2.0))
η = getnaturalparameters(ef)

f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(VonMisesFisher, η))
autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
autograd_information(η)

λ = 0.2

dist = Geometric(λ)
ef = convert(KnownExponentialFamilyDistribution, dist)

η = getnaturalparameters(ef)

f_logpartition = (η) -> logpartition(KnownExponentialFamilyDistribution(Geometric, η))
autograd_information = (η) -> Zygote.hessian(f_logpartition, η)
autograd_information(η)

fisherinformation(ef)

fisherinformation(dist)

n_samples = 10000
samples = rand(Geometric(λ), n_samples)

totalHessian = zeros(typeof(λ), 1, 1)
for sample in samples
    totalHessian -= ForwardDiff.hessian((params) -> logpdf.(Geometric(params[1]), sample), [λ])
end

first(totalHessian) / n_samples

transformation(η) = one(Float64) - exp(η[1])

J = ForwardDiff.jacobian(transformation, η)
@test J'*fisherinformation(dist)*J ≈ fisherinformation(ef) atol = 1e-8