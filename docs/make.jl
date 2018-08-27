using Documenter, IRAM

makedocs(
	modules = [IRAM],
	format = :html,
	doctest = false,
	clean = true,
	sitename = "IRAM.jl",
	pages = [
		"Home" => "index.md",
		"Theory" => "theory.md",
		"Using IRAM.jl" => [
			"Getting started" => "usage/01_getting_started.md",
			"Transformations" => "usage/02_spectral_transformations.md"
		]
	]
)

deploydocs(
	repo = "github.com/haampie/IRAM.jl.git",
	target = "build",
	osname = "linux",
	julia  = "0.7",
	deps = nothing,
	make = nothing,
)
