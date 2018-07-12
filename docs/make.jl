using Documenter, IRAM

makedocs(
	modules = [IRAM],
	format = :html,
	doctest = false,
	clean = true,
	sitename = "IRAM.jl",
	pages = [
		"Home" => "index.md",
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
