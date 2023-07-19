using Distributions, BenchmarkTools, FillArrays
import ExponentialFamily:MatrixDirichlet, KnownExponentialFamilyDistribution,getnaturalparameters,pack_naturalparameters,unpack_naturalparameters,logpartition,basemeasure,sufficientstatistics,fisherinformation
import StatsFuns: logistic,softmax
dist = MatrixDirichlet([1 1; 1 1])
ef = convert(KnownExponentialFamilyDistribution, dist)
@btime pack_naturalparameters($dist)
@btime unpack_naturalparameters($ef)
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

ef1 = KnownExponentialFamilyDistribution(Categorical, log.([1.0,  1.0]))

dist1 = convert(Distribution,ef1)
getnaturalparameters(ef1)
logpdf(ef1,2)
logpdf(dist1,2)
pack_naturalparameters(dist)
using FillArrays


OneElement(1,10)
using BlockArrays

BlockArray{Int64}(zeros(4,4), [2,2],[2,2])

@btime fisherinformation($ef)
@btime fisherinformation($dist)
@btime unpack_naturalparameters($ef) + Ones{Float64}(2,2) 