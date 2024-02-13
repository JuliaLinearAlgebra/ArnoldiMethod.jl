using Documenter, ArnoldiMethod

makedocs(
    modules = [ArnoldiMethod],
    sitename = "ArnoldiMethod.jl",
    format = Documenter.HTML(),
    pages = ["Home" => "index.md"],
    warnonly = [:missing_docs, :cross_references],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JuliaLinearAlgebra/ArnoldiMethod.jl.git",
    devbranch = "master",
)
