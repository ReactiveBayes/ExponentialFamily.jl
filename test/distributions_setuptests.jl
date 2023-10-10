
using Test, ExponentialFamily, Distributions, StaticArrays, StableRNGs, LinearAlgebra, FillArrays

import Distributions: variate_form, value_support

import ExponentialFamily:
    deep_eltype,
    sampletype,
    samplefloattype,
    promote_sampletype,
    promote_samplefloattype,
    paramfloattype,
    convert_paramfloattype,
    FactorizedJoint,
    PromoteTypeConverter

function generate_random_distributions(::Type{V} = Any; seed = abs(rand(Int)), Types = (Float32, Float64)) where {V}
    rng = StableRNG(seed)
    distributions = []

    # Add `Univariate` distributions
    for T in Types
        push!(distributions, NormalMeanPrecision(rand(rng, T), rand(rng, T)))
        push!(distributions, NormalMeanVariance(rand(rng, T), rand(rng, T)))
        push!(distributions, NormalWeightedMeanPrecision(rand(rng, T), rand(rng, T)))
        push!(distributions, GammaShapeRate(rand(rng, T), rand(rng, T)))
        push!(distributions, GammaShapeScale(rand(rng, T), rand(rng, T)))
    end

    # Add `Multivariate` distributions
    for T in Types, n in (2, 3)
        push!(distributions, MvNormalMeanPrecision(rand(rng, T, n)))
        push!(distributions, MvNormalMeanCovariance(rand(rng, T, n)))
        push!(distributions, MvNormalWeightedMeanPrecision(rand(rng, T, n)))
    end

    # Add `Matrixvariate` distributions
    for T in Types, n in (2, 3)
        push!(distributions, InverseWishart(5one(T), Array{T}(Eye(n))))
        push!(distributions, Wishart(5one(T), Array{T}(Eye(n))))
    end

    return filter((dist) -> variate_form(typeof(dist)) <: V, distributions)
end
