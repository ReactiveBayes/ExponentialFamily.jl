using Documenter
using ExponentialFamily
using Distributions

DocMeta.setdocmeta!(ExponentialFamily, :DocTestSetup, :(using ExponentialFamily, Distributions); recursive=true)

makedocs(
    modules  = [ ExponentialFamily ],
    clean    = true,
    sitename = "ExponentialFamily.jl",
    strict   = [ :doctest, :eval_block, :example_block, :meta_block, :parse_error, :setup_block ],
    pages    = [
        "Home"      => "index.md",
        "Interface" => "interface.md",
        "Library"   => "library.md",
        "Examples"  => "examples.md",
    ],

    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),

)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/biaslab/ExponentialFamily.jl.git"
    )
end