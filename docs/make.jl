using Documenter, ArnoldiMethod

makedocs(
    modules = [ArnoldiMethod],
    sitename = "ArnoldiMethod.jl",
    format = Documenter.HTML(),
    pages = [
	"Home" => "index.md",
	"Theory" => "theory.md",
	"Using ArnoldiMethod.jl" => [
	    "Getting started" => "usage/01_getting_started.md",
	    "Transformations" => "usage/02_spectral_transformations.md"
	]
    ],
    warnonly = [:missing_docs, :cross_references],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JuliaLinearAlgebra/ArnoldiMethod.jl.git",
    devbranch="main"
)
