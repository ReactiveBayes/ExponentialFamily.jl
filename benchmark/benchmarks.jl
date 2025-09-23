using BenchmarkTools
using ExponentialFamily, Distributions, Random
using ExponentialFamily.BayesBase

const SUITE = BenchmarkGroup()

include("benchmarks/bernoulli.jl")
include("benchmarks/dirichlet_collection.jl")
include("benchmarks/normal_family/mv_normal_mean_covariance.jl")