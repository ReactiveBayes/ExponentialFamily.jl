using Documenter
using ExponentialFamily

makedocs(
    modules  = [ ExponentialFamily ],
    clean    = true,
    sitename = "ExponentialFamily.jl",
    pages    = [
        "Home"      => "index.md",
        "Methods"   => "methods.md",
        "Examples"  => "examples.md",
    ],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/biaslab/ExponentialFamily.jl.git"
    )
end