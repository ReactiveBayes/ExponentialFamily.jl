using BenchmarkTools
using ExponentialFamily, Distributions, Random
using ExponentialFamily.BayesBase

const SUITE = BenchmarkGroup()

include("benchmarks/bernoulli.jl")
include("benchmarks/tensordirichlet.jl")