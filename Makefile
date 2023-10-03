scripts_init:
	julia --startup-file=no --project=scripts/ -e 'using Pkg; Pkg.resolve(); Pkg.update(); Pkg.precompile();'

lint: scripts_init ## Code formating check
	julia --startup-file=no --project=scripts/ scripts/format.jl

format: scripts_init ## Code formating run
	julia --startup-file=no --project=scripts/ scripts/format.jl --overwrite

benchmark: scripts_init ## Code formating run
	julia --startup-file=no --project=scripts/ scripts/benchmark.jl

.PHONY: docs

doc_init:
	julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate();'

docs: doc_init ## Generate documentation
	julia --project=docs/ docs/make.jl

servedocs: doc_init ## Serve documentation (and auto-reload on change), requires LiveServer.jl
	julia --project=docs -e 'using LiveServer; servedocs()'