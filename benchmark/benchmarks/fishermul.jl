using LinearAlgebra
using ExponentialFamily: WishartFast, InverseWishartFast

SUITE["fishermul"] = BenchmarkGroup()

function add_fishermul_benchmarks(label, d)
    ef = convert(ExponentialFamilyDistribution, d)
    v = randn(MersenneTwister(72), length(getnaturalparameters(ef)))
    w = similar(v)
    buffer = FisherMulBuffer(ef, v)
    group = SUITE["fishermul"][label] = BenchmarkGroup()
    group["matrix"] = @benchmarkable fisherinformation($ef) * $v
    group["returning"] = @benchmarkable fishermul($ef, $v)
    group["inplace"] = @benchmarkable fishermul!($w, $ef, $v)
    group["buffered"] = @benchmarkable fishermul!($w, $ef, $v, $buffer)
    return nothing
end

for n in (16, 64, 256)
    add_fishermul_benchmarks("Categorical/$n", Categorical(fill(1 / n, n)))
    add_fishermul_benchmarks("Dirichlet/$n", Dirichlet(fill(2.0, n)))
    add_fishermul_benchmarks("DirichletCollection/$n", DirichletCollection(fill(2.0, n, 4)))
    add_fishermul_benchmarks("VonMisesFisher/$n", VonMisesFisher(fill(inv(sqrt(n)), n), 4.0))
    add_fishermul_benchmarks("MvNormalMeanScalePrecision/$n", MvNormalMeanScalePrecision(fill(0.3, n), 2.0))
end

for n in (2, 4, 8)
    rng = MersenneTwister(n)
    A = randn(rng, n, n)
    S = A * A' + n * I
    μ = randn(rng, n)
    M = randn(rng, n, 2)
    V = [2.0 0.2; 0.2 1.5]
    for d in (MvNormalMeanCovariance(μ, S), WishartFast(n + 4.0, S),
        InverseWishartFast(n + 4.0, S), MvNormalGamma(μ, S, 3.0, 2.0),
        MvNormalWishart(μ, S, 2.0, n + 4.0), MatrixNormal(M, S, V),
        MatrixNormalWishart(M, S, V, 5.0))
        tag = ExponentialFamily.exponential_family_typetag(d)
        add_fishermul_benchmarks("$tag/$n", d)
    end
end
