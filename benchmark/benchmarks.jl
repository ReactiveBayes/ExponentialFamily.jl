using BenchmarkTools
using ExponentialFamily, Distributions, Random

const SUITE = BenchmarkGroup()

include("benchmarks/bernoulli.jl")