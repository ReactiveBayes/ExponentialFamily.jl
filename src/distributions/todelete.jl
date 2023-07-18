using Distributions, BenchmarkTools
import ExponentialFamily: KnownExponentialFamilyDistribution,getnaturalparameters,pack_naturalparameters,unpack_naturalparameters

dist = Bernoulli(0.9)
ef = KnownExponentialFamilyDistribution(Bernoulli,[0.1])
@btime pack_naturalparameters($dist)
@btime unpack_naturalparameters($ef)