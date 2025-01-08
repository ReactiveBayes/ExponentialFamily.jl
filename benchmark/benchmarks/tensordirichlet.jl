
SUITE["tensordirichlet"] = BenchmarkGroup(
    ["tensordirichlet", "distribution"],
    "prod" => BenchmarkGroup(["prod", "multiplication"]),
    "convert" => BenchmarkGroup(["convert"])
)

# `prod` BenchmarkGroup ========================
for rank in (3, 5, 10)
    for d in (2, 5, 10, 20)
        left = TensorDirichlet(rand([d for _ in 1:rank]...))
        right = TensorDirichlet(rand([d for _ in 1:rank]...))
        SUITE["tensordirichlet"]["prod"]["Closed(rank=$rank, d=$d)"] = @benchmarkable prod(ClosedProd(), $left, $right)
    end
end

# ==============================================

# # `convert` BenchmarkGroup =====================
# SUITE["tensordirichlet"]["convert"]["Convert from D to EF"] = @benchmarkable convert(ExponentialFamilyDistribution, dist) setup = begin
#     dist = TensorDirichlet(rand(5, 5, 5))
# end

# SUITE["tensordirichlet"]["convert"]["Convert from EF to D"] = @benchmarkable convert(Distribution, efdist) setup = begin
#     efdist = convert(ExponentialFamilyDistribution, TensorDirichlet(rand(5, 5, 5)))
# end
# ==============================================

for rank in (3, 5)
    for d in (2, 5, 10, 20)
        distribution = TensorDirichlet(rand([d for _ in 1:rank]...))
        sample = rand(distribution)
        SUITE["tensordirichlet"]["mean"]["rank=$rank, d=$d"] = @benchmarkable mean($distribution)
        SUITE["tensordirichlet"]["rand"]["rank=$rank, d=$d"] = @benchmarkable rand($distribution)
        SUITE["tensordirichlet"]["logpdf"]["rank=$rank, d=$d"] = @benchmarkable logpdf($distribution, $sample)
        SUITE["tensordirichlet"]["var"]["rank=$rank, d=$d"] = @benchmarkable var($distribution)
        SUITE["tensordirichlet"]["cov"]["rank=$rank, d=$d"] = @benchmarkable cov($distribution)
    end
end

# ==============================================