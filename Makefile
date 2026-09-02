.PHONY: help docs docs-serve docs-clean test format check-format clean deps deps-docs deps-scripts benchmark benchmark-compare


DOCSRC = docs
DOCTARGET = $(DOCSRC)/build

SCRIPTSRC = scripts
FORMATTER = $(SCRIPTSRC)/format.jl
BENCHMARK = $(SCRIPTSRC)/benchmark.jl

# `--startup-file=no` keeps an error in a contributor's `~/.julia/config/startup.jl`
# from taking down unrelated targets such as `make format` with a confusing stacktrace
JULIA ?= julia --startup-file=no
JULIAFLAGS ?= --project=.
JULIAFLAGSDOCS ?= --project=$(DOCSRC)
JULIAFLAGSSCRIPTS ?= --project=$(SCRIPTSRC)

# Colors for terminal output
ifdef NO_COLOR
GREEN  :=
YELLOW :=
WHITE  :=
RESET  :=
else
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)
endif

# Default target
.DEFAULT_GOAL := help

## Show help for each of the Makefile targets
help:
	@echo ''
	@echo 'ExponentialFamily.jl Makefile ${YELLOW}targets${RESET}:'
	@echo ''
	@echo '${GREEN}Documentation commands:${RESET}'
	@echo '  ${YELLOW}docs${RESET}                 Build the documentation'
	@echo '  ${YELLOW}docs-init${RESET}            Install documentation requirements'
	@echo '  ${YELLOW}docs-serve${RESET}           Serve documentation locally for preview in browser'
	@echo '  ${YELLOW}docs-clean${RESET}           Clean the documentation build directory'
	@echo ''
	@echo '${GREEN}Development commands:${RESET}'
	@echo '  ${YELLOW}deps${RESET}                 Install project dependencies'
	@echo '  ${YELLOW}deps-docs${RESET}            Install documentation dependencies'
	@echo '  ${YELLOW}deps-scripts${RESET}         Install script dependencies'
	@echo '  ${YELLOW}test${RESET}                 Run project tests (test_args="path1 path2" runs a subset)'
	@echo '  ${YELLOW}format${RESET}               Format Julia code'
	@echo '  ${YELLOW}check-format${RESET}         Check Julia code formatting (does not modify files)'
	@echo '  ${YELLOW}clean${RESET}                Clean all generated files'
	@echo ''
	@echo '${GREEN}Benchmark commands:${RESET}'
	@echo '  ${YELLOW}benchmark${RESET}            Run project benchmarks'
	@echo '  ${YELLOW}benchmark-compare${RESET}    Run project benchmarks with comparison against specified branch'
	@echo ''
	@echo '${GREEN}Help:${RESET}'
	@echo '  ${YELLOW}help${RESET}                 Show this help message'
	@echo ''
	@echo '${GREEN}Environment variables:${RESET}'
	@echo '  ${YELLOW}NO_COLOR${RESET}             Set this variable to any value to disable colored output'
	@echo ''

## Documentation commands:
docs: deps-docs ## Build the documentation
	$(JULIA) $(JULIAFLAGSDOCS) docs/make.jl

docs-init: deps-docs ## Serve documentation locally for preview in browser
	$(JULIA) $(JULIAFLAGSDOCS) -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'

docs-serve: deps-docs ## Serve documentation locally for preview in browser
	$(JULIA) $(JULIAFLAGSDOCS) -e 'using LiveServer; LiveServer.servedocs(launch_browser=true, port=5678)'

docs-clean: ## Clean the documentation build directory
	rm -rf $(DOCTARGET)

## Development commands:
deps: ## Install project dependencies
	$(JULIA) $(JULIAFLAGS) -e 'using Pkg; Pkg.instantiate()'

deps-docs: ## Install documentation dependencies
	$(JULIA) $(JULIAFLAGSDOCS) -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

deps-scripts: ## Install script dependencies
	$(JULIA) $(JULIAFLAGSSCRIPTS) -e 'using Pkg; Pkg.instantiate()'

# `test_args` selects a subset of the suite; each argument is a path to a test file or
# directory, resolved against the package root and then `test/`. `Aqua` is skipped for such
# runs unless `RUN_AQUA=true` is set. See `test/runtests.jl`.
test: deps ## Run project tests (test_args="path1 path2" runs a subset)
	$(JULIA) $(JULIAFLAGS) -e 'using Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

# `JuliaFormatter` is pinned in `scripts/Project.toml`. Its output changes between
# minor releases and `scripts/Manifest.toml` is gitignored, so leaving it unbounded
# meant every contributor resolved whatever version was newest when they first ran
# this target, and two people formatting the same untouched file got different diffs.
# It also parses through `JuliaSyntax`, so its output can shift with the Julia minor
# version independently of the formatter version -- which is why the CI job that runs
# `check-format` pins its Julia version to the top of the test matrix.
# Bump the pin intentionally and run `make format` over the whole repo in the same commit.
format: deps-scripts ## Format Julia code
	$(JULIA) $(JULIAFLAGSSCRIPTS) $(FORMATTER) --overwrite

check-format: deps-scripts ## Check Julia code formatting (does not modify files)
	$(JULIA) $(JULIAFLAGSSCRIPTS) $(FORMATTER)

benchmark: deps-scripts ## Run project benchmarks
	$(JULIA) $(JULIAFLAGSSCRIPTS) $(JULIAFLAGSSCRIPTS) $(BENCHMARK)

benchmark-compare: deps-scripts ## Run project benchmarks with comparison against specified branch
	$(JULIA) $(JULIAFLAGSSCRIPTS) $(JULIAFLAGSSCRIPTS) $(BENCHMARK) $(branch)

clean: docs-clean ## Clean all generated files
	rm -rf .julia/compiled
