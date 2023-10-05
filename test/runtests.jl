using CpuId, ReTestItems, ExponentialFamily

runtests(ExponentialFamily,
    nworkers = cpucores(),
    nworker_threads = Int(cputhreads() / cpucores()),
    memory_threshold = 1.0
)
