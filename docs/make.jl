using Documenter, IRAM

makedocs(
	modules = [IRAM],
	format = :html,
	doctest = false,
	clean = true,
	sitename = "IRAM.jl",
	pages = [
		"Home" => "index.md",
		"Theory" => [
			"Eigenvalue problems" => "theory/transformations.md",
			"Schur decomposition" => "theory/partial_schur.md"
		],
		"Using IRAM.jl" => "usage/usage.md"
	]
)

deploydocs(
	repo = "github.com/haampie/IRAM.jl.git",
	target = "build",
	osname = "linux",
	julia  = "0.6",
	deps = nothing,
	make = nothing,
)
