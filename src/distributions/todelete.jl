using Distributions, BenchmarkTools
import ExponentialFamily: KnownExponentialFamilyDistribution,getnaturalparameters,pack_naturalparameters,unpack_naturalparameters,logpartition,basemeasure,sufficientstatistics
import StatsFuns: logistic
dist = Beta(1, 1.2)
ef = convert(KnownExponentialFamilyDistribution, dist)
@btime pack_naturalparameters($dist)
@btime unpack_naturalparameters($ef)
@code_warntype logpartition(ef)
@code_lowered logpartition(ef)
@btime logpartition($ef)
@btime basemeasure($ef, 0.1)
@btime sufficientstatistics($ef,$0.1)
@btime logpdf($ef,$0.1)
@btime logpdf($dist,$0.1)
using StaticArrays
dot([1], SA[1])
@btime log(logistic(η))