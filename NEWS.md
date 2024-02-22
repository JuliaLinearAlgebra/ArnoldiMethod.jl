# ArnoldiMethod.jl v0.4.0

- BREAKING: The target structs `LM`, `SR`, `LR`, `SI`, `LI` are no longer
  exported. You have to import them explicitly: `using ArnoldiMethod: LM`. But
  easier it to update your code from structs `LM()` to symbols `:LM`.

# ArnoldiMethod.jl v0.3.5

- ArnoldiMethod.jl now exports `partialschur!` and `ArnoldiWorkspace`, which
  can be used to pre-allocate (custom) matrices. Also it can be used to run the
  algorithm from an existing partial Schur decomposition.

# ArnoldiMethod.jl v0.3.0

- The required Julia version is now 1.6 or higher
