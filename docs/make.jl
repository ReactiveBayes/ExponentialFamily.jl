using Documenter
using ExponentialFamily
using Distributions

DocMeta.setdocmeta!(ExponentialFamily, :DocTestSetup, :(using ExponentialFamily, Distributions, BayesBase); recursive = true)

makedocs(
    modules  = [ExponentialFamily],
    clean    = true,
    sitename = "ExponentialFamily.jl",
    pages    = [
    "Home"      => "index.md",
    "Getting Started" => [
        "What is a Probability Distribution?" => "distributions.md",
        "What is the Exponential Family?"     => "exponential_family.md",
        "Comparison with Distributions.jl"    => "comparison.md"
    ],
    "Interface" => "interface.md",
    "Library"   => "library.md",
    "Examples"  => "examples.md"
    ],
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true")
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/ReactiveBayes/ExponentialFamily.jl.git",
        devbranch = "main",
        forcepush = true
    )
end
