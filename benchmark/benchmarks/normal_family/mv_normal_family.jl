using LinearAlgebra
using StaticArrays
using ExponentialFamily.BayesBase
using StableRNGs

SUITE["mvnormal_family"] = BenchmarkGroup(
    ["normal_family", "distribution", "logpartition"],
    "gradlogpartition" => BenchmarkGroup(["gradlogpartition"])
)

union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type)  = (x,)

# prod (PreserveType) ==============
let dims_dense = (10, 50, 100, 500)
    # Dense × Dense 
    rng = StableRNG(42)
    for d in dims_dense
        μ = 10randn(d)
        L = LowerTriangular(randn(d, d) + d * I)
        Σ = L * L'
        ef = convert(ExponentialFamilyDistribution, MvNormalMeanCovariance(μ, Σ))
        SUITE["mvnormal_family"]["logpartition"]["d=$d"] =
            @benchmarkable logpartition($ef) 
        SUITE["mvnormal_family"]["gradlogpartition"]["d=$d"] =
            @benchmarkable gradlogpartition($ef) 
    end
end
