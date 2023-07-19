using Distributions, BenchmarkTools, FillArrays
import ExponentialFamily: KnownExponentialFamilyDistribution,getnaturalparameters,pack_naturalparameters,unpack_naturalparameters,logpartition,basemeasure,sufficientstatistics
import StatsFuns: logistic,softmax
dist = Categorical([1/2, 1/3, 1/6])
ef = convert(KnownExponentialFamilyDistribution, dist)
@btime pack_naturalparameters($dist)
@btime getnaturalparameters($ef)
# @btime unpack_naturalparameters($ef)
@code_warntype logpartition(ef)
@code_lowered logpartition(ef)
@btime logpartition($ef)

@btime basemeasure($ef, $2)
@btime sufficientstatistics($ef,$2)
@btime pdf($ef,$3)
@btime pdf($dist, $3)
using StaticArrays
dot([1], SA[1])
@btime log(logistic(η))
using SparseArrays
@btime Vector{Int64}(undef,10)
@btime convert(Distribution,ef)
@btime logpdf2($ef,$2)

ef1 = KnownExponentialFamilyDistribution(Categorical, log.([1.0,  1.0]))

dist1 = convert(Distribution,ef1)
getnaturalparameters(ef1)
logpdf(ef1,2)
logpdf(dist1,2)
pack_naturalparameters(dist)
using FillArrays


OneElement(1,10)
using BlockArrays

BlockArray{Int64}(undef_blocks, [2,2], [2,2])