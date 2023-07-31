using Distributions, BenchmarkTools, FillArrays
using ExponentialFamily
import ExponentialFamily:ContinuousBernoulli, MvNormalMeanCovariance,NormalGamma,Contingency, MatrixDirichlet, ExponentialFamilyDistribution,getnaturalparameters,pack_naturalparameters,unpack_naturalparameters,logpartition,basemeasure,sufficientstatistics,fisherinformation, Unsafe,support
import ExponentialFamily: isproper
dist = Bernoulli(0.5)
ef = convert(ExponentialFamilyDistribution, dist)
@btime convert(Distribution, $ef)
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
@btime pdf($ef,$1)
@btime fisherinformation($dist)
@btime pdf($dist, $1)
@btime pdf($MvNormal([$1,2], $[0.1 -0.2;-0.2 0.9]),$[1,2])


