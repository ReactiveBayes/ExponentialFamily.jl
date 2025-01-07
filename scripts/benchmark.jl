using PkgBenchmark, BenchmarkTools, Dates

import ExponentialFamily

mkpath("./benchmark_logs")

if isempty(ARGS)
    result = PkgBenchmark.benchmarkpkg(ExponentialFamily)
    export_markdown("./benchmark_logs/benchmark_$(now()).md", result)
    export_markdown("./benchmark_logs/last.md", result)
else
    name = first(ARGS)
    BenchmarkTools.judge(ExponentialFamily, name; judgekwargs = Dict(:time_tolerance => 0.1, :memory_tolerance => 0.05))
    export_markdown("benchmark_vs_$(name)_result.md", result)
end
