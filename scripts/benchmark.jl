using PkgBenchmark, Dates

import ExponentialFamily

results = benchmarkpkg(ExponentialFamily)

mkpath("./benchmark_logs")

export_markdown("./benchmark_logs/benchmark_$(now()).md", results)
export_markdown("./benchmark_logs/last.md", results)
