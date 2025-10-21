.PHONY: help docs docs-serve docs-clean test format check-format clean deps deps-docs deps-scripts benchmark benchmark-compare


DOCSRC = docs
DOCTARGET = $(DOCSRC)/build

SCRIPTSRC = scripts
FORMATTER = $(SCRIPTSRC)/format.jl
BENCHMARK = $(SCRIPTSRC)/benchmark.jl

JULIA ?= julia
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
	@echo 'PLACEHOLDERNAME_CHANGE_MAKEFILE_LINE_22.jl Makefile ${YELLOW}targets${RESET}:'
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
	@echo '  ${YELLOW}test${RESET}                 Run project tests'
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

test: deps ## Run project tests
	$(JULIA) $(JULIAFLAGS) -e 'using Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'	

format: deps-scripts ## Format Julia code
	$(JULIA) $(JULIAFLAGSSCRIPTS) $(FORMATTER) --overwrite

check-format: deps-scripts ## Check Julia code formatting (does not modify files)
	$(JULIA) $(JULIAFLAGSSCRIPTS) $(FORMATTER)

benchmark: deps-scripts ## Run project benchmarks
	$(JULIA) $(JULIAFLAGSSCRIPTS) $(JULIAFLAGSSCRIPTS) $(BENCHMARK)

benchmark-compare: deps-scripts ## Run project benchmarks with comparison against specified branch
	$(JULIA) $(JULIAFLAGSSCRIPTS) $(JULIAFLAGSSCRIPTS) $(BENCHMARK) --compare-branch $(branch)

clean: docs-clean ## Clean all generated files
	rm -rf .julia/compiled
