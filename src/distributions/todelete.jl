using Distributions, BenchmarkTools, FillArrays
using ExponentialFamily
import ExponentialFamily:ContinuousBernoulli, MvNormalMeanCovariance,NormalGamma,Contingency, MatrixDirichlet, ExponentialFamilyDistribution,getnaturalparameters,pack_naturalparameters,unpack_naturalparameters,logpartition,basemeasure,sufficientstatistics,fisherinformation, Unsafe,support
import ExponentialFamily: isproper
dist = Bernoulli(0.5)
ef = convert(ExponentialFamilyDistribution, dist)
@btime convert(ExponentialFamilyDistribution, $dist)
# ef = ExponentialFamilyDistribution(LogNormal,[22,-3],nothing,Unsafe())
@btime pack_naturalparameters($dist)
@btime unpack_naturalparameters($ef)
# @btime unpack_naturalparameters($ef)
@code_warntype logpartition(ef)
@code_lowered logpartition(ef)
@btime logpartition($ef)
isproper(ef)
ExponentialFamily.support(ef)
@btime ExponentialFamily.insupport($ef, $0.2)
@btime basemeasure($ef)
@btime sufficientstatistics(Bernoulli)(1)
@code_warntype logpdf(ef, 1)
@btime logpdf($ef,$1)
@btime fisherinformation($dist)
@btime logpdf(dist, 1)
@btime pdf($MvNormal([$1,2], $[0.1 -0.2;-0.2 0.9]),$[1,2])
using StaticArrays
dot([1], SA[1])
@btime log(logistic(η))
using SparseArrays
@btime Vector{Int64}(undef,10)
@btime convert(Distribution,ef)

ef1 = ExponentialFamilyDistribution(MatrixDirichlet, log.([1.0,  1.0, 1.0, 1.0]))

dist1 = convert(Distribution,ef1)
getnaturalparameters(ef1)
@btime logpdf($ef1,$2)
@btime fisherinformation($ef1)
logpdf(dist1,2)
pack_naturalparameters(dist)
using FillArrays


OneElement(1,10)
using BlockArrays

BlockArray{Int64}(zeros(4,4), [2,2],[2,2])

@btime fisherinformation($ef)
@btime fisherinformation($dist)
@btime unpack_naturalparameters($ef) + Ones{Float64}(2,2) 

using LogExpFunctions

A = [1 0; 0 1]
logsumexp.(A)

using SparseArrays, LinearAlgebra, FillArrays
@btime Diagonal([1, 2])
x = [1.0, 2.0]
@btime deepcopy(x)
@btime map!(log,x,x)

using StaticArrays, LoopVectorization

x = SA[4,2, 3, 3]
sort(x)
vcat([1],[2])

@btime prod(factorial.($[1, 2, 3]))
@btime prod(map(factorial, $[1,2,3]))

@btime @.factorial($[1,2,3])

@btime exp.($[1,2,3])
@btime @.exp($[1,2,3])
@btime vmap(exp,$[1,2,3])
@btime vmap(exp, $[1,2,3])
OneElement(1, (1,1),(5,5))

@btime sum($[1,2,3])
@btime vreduce(+, $[1,2,3])

Fill(d -> exp, [1,2,3], (3,3))

Distributions.sqmahal([1 0; 0 1],zeros(2))
@btime vcat($[1,0],view($[2 1; 1 0],:))