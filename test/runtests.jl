using Aqua, Hwloc, ReTestItems, ExponentialFamily

# Allow selecting a subset of the suite from the command line, e.g.
#   make test test_args="test/distributions/beta_tests.jl"
# Each argument is a path to a test file or directory. `:` is accepted as a separator
# (the ReactiveMP.jl convention, e.g. `distributions:beta_tests.jl`) and paths are resolved
# against the package root and then `test/`, so that filtering works regardless of the
# working directory `Pkg.test` runs in and regardless of which of the two spellings is used.
const TESTPATHS = map(ARGS) do arg
    relative = joinpath(split(arg, ':')...)
    candidates = isabspath(relative) ? (relative,) : (joinpath(dirname(@__DIR__), relative), joinpath(@__DIR__, relative))
    index = findfirst(ispath, candidates)
    isnothing(index) && error("Cannot find the test path `$(arg)`. Tried: $(join(candidates, ", ")).")
    return candidates[index]
end

# `Aqua` takes a significant part of the total runtime, which defeats the purpose of running a
# subset of the suite, so it is skipped whenever `test_args` selects one. `RUN_AQUA` overrides
# the default in both directions (the ReactiveMP.jl convention).
# `ambiguities = false` - there are quite some ambiguities, but these should be normal and should not be encountered under normal circumstances
# `piracies = false` - we extend/add some of the methods to the objects defined in the Distributions.jl
if get(ENV, "RUN_AQUA", isempty(TESTPATHS) ? "true" : "false") == "true"
    Aqua.test_all(ExponentialFamily, ambiguities = false, deps_compat = (; check_extras = false, check_weakdeps = true), piracies = false)
end

ncores = max(Hwloc.num_physical_cores(), 1)
nthreads = max(Hwloc.num_virtual_cores(), 1)
threads_per_core = max(Int(floor(nthreads / ncores)), 1)

# `runtests` accepts either the package itself (the whole suite) or a list of paths.
targets = isempty(TESTPATHS) ? (ExponentialFamily,) : TESTPATHS

runtests(targets...,
    nworkers = ncores,
    nworker_threads = threads_per_core,
    memory_threshold = 1.0
)
