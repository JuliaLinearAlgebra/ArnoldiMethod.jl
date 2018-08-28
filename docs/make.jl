using Documenter, ArnoldiMethod

makedocs(
	modules = [ArnoldiMethod],
	format = :html,
	doctest = false,
	clean = true,
	sitename = "ArnoldiMethod.jl",
	pages = [
		"Home" => "index.md",
		"Theory" => "theory.md",
		"Using ArnoldiMethod.jl" => [
			"Getting started" => "usage/01_getting_started.md",
			"Transformations" => "usage/02_spectral_transformations.md"
		]
	]
)

deploydocs(
	repo = "github.com/haampie/ArnoldiMethod.jl.git",
	target = "build",
	osname = "linux",
	julia  = "0.7",
	deps = nothing,
	make = nothing,
)
