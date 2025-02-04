SUITE["dirichlet_collection"] = BenchmarkGroup(
    ["dirichlet_collection", "distribution"],
    "prod" => BenchmarkGroup(["prod", "multiplication"]),
    "convert" => BenchmarkGroup(["convert"])
)

# `prod` BenchmarkGroup ========================
for rank in (3, 4, 5, 6)
    for d in (5, 10, 20)
        left = DirichletCollection(rand([d for _ in 1:rank]...) .+ 1)
        right = DirichletCollection(rand([d for _ in 1:rank]...) .+ 1)
        SUITE["dirichlet_collection"]["prod"]["Closed(rank=$rank, d=$d)"] = @benchmarkable prod(ClosedProd(), $left, $right)
    end
end

# `convert` BenchmarkGroup =====================
SUITE["dirichlet_collection"]["convert"]["Convert from D to EF"] = @benchmarkable convert(ExponentialFamilyDistribution, dist) setup = begin
    dist = DirichletCollection(rand(5, 5, 5))
end

SUITE["dirichlet_collection"]["convert"]["Convert from EF to D"] = @benchmarkable convert(Distribution, efdist) setup = begin
    efdist = convert(ExponentialFamilyDistribution, DirichletCollection(rand(5, 5, 5)))
end

for rank in (3, 4, 5, 6)
    for d in (5, 10, 20)
        distribution = DirichletCollection(rand([d for _ in 1:rank]...))
        sample = rand(distribution)
        SUITE["dirichlet_collection"]["mean"]["rank=$rank, d=$d"] = @benchmarkable mean($distribution)
        SUITE["dirichlet_collection"]["rand"]["rank=$rank, d=$d"] = @benchmarkable rand($distribution)
        SUITE["dirichlet_collection"]["logpdf"]["rank=$rank, d=$d"] = @benchmarkable logpdf($distribution, $sample)
        SUITE["dirichlet_collection"]["var"]["rank=$rank, d=$d"] = @benchmarkable var($distribution)
        SUITE["dirichlet_collection"]["cov"]["rank=$rank, d=$d"] = @benchmarkable cov($distribution)
        SUITE["dirichlet_collection"]["entropy"]["rank=$rank, d=$d"] = @benchmarkable entropy($distribution)
    end
end