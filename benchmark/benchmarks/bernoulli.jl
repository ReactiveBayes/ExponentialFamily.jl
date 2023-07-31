
SUITE["bernoulli"] = BenchmarkGroup(
    ["ber", "bernoulli", "distribution"],
    "prod" => BenchmarkGroup(["prod", "multiplication"]),
    "convert" => BenchmarkGroup(["convert"])
)

# `prod` BenchmarkGroup ========================
SUITE["bernoulli"]["prod"]["Closed"] = @benchmarkable prod(ClosedProd(), left, right) setup = begin
    left, right = Bernoulli(0.5), Bernoulli(0.5)
end
# ==============================================


# `convert` BenchmarkGroup =====================
SUITE["bernoulli"]["convert"]["Convert from D to EF"] = @benchmarkable convert(ExponentialFamilyDistribution, dist) setup = begin
    dist = Bernoulli(0.5)
end

SUITE["bernoulli"]["convert"]["Convert from EF to D"] = @benchmarkable convert(Distribution, efdist) setup = begin
    efdist = convert(ExponentialFamilyDistribution, Bernoulli(0.5))
end
# ==============================================