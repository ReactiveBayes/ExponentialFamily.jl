using Aqua, CpuId, ReTestItems, ExponentialFamily

# `ambiguities = false` - there are quite some ambiguities, but these should be normal and should not be encountered under normal circumstances
# `piracy = false` - we extend/add some of the methods to the objects defined in the Distributions.jl
Aqua.test_all(ExponentialFamily, ambiguities = false, piracy = false)

runtests(ExponentialFamily,
    nworkers = cpucores(),
    nworker_threads = Int(cputhreads() / cpucores()),
    memory_threshold = 1.0
)
