# ArnoldiMethod.jl v0.3.0

- ArnoldiMethod.jl now exports `partialschur!` and `ArnoldiWorkspace`, which can be used to
  pre-allocate (custom) matrices. Also it can be used to run the algorithm from an existing partial
  Schur decomposition.


- The required Julia version is now 1.6 or higher
