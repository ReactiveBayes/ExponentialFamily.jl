using Aqua, Hwloc, ReTestItems, ExponentialFamily

# `ambiguities = false` - there are quite some ambiguities, but these should be normal and should not be encountered under normal circumstances
# `piracies = false` - we extend/add some of the methods to the objects defined in the Distributions.jl
Aqua.test_all(ExponentialFamily, ambiguities = false, deps_compat = (; check_extras = false, check_weakdeps = true), piracies = false)

ncores = max(Hwloc.num_physical_cores(), 1)
nthreads = max(Hwloc.num_virtual_cores(), 1)
threads_per_core = max(Int(floor(nthreads / ncores)), 1)

# Allow selecting a subset of test files/directories from the command line, e.g.
# `make test test_args="test/distributions/beta_tests.jl"`. The `:` separator (as used in
# ReactiveMP.jl, e.g. `distributions:beta_tests.jl`) is mapped to path separators.
if isempty(ARGS)
    runtests(ExponentialFamily,
        nworkers = ncores,
        nworker_threads = threads_per_core,
        memory_threshold = 1.0
    )
else
    # Resolve the requested paths relative to the package root so that filtering works
    # regardless of the working directory `Pkg.test` runs in.
    pkgroot = dirname(@__DIR__)
    paths = map(ARGS) do arg
        p = joinpath(split(arg, ":")...)
        isabspath(p) ? p : joinpath(pkgroot, p)
    end
    runtests(paths...,
        nworkers = ncores,
        nworker_threads = threads_per_core,
        memory_threshold = 1.0
    )
end
