# ArnoldiMethod.jl

[![CI](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl/actions/workflows/ci.yml) [![Codecov](https://codecov.io/github/JuliaLinearAlgebra/ArnoldiMethod.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaLinearAlgebra/ArnoldiMethod.jl?branch=master)



The Arnoldi Method with Krylov-Schur restart, natively in Julia.

## Docs
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://julialinearalgebra.github.io/ArnoldiMethod.jl/stable) [![Latest docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://julialinearalgebra.github.io/ArnoldiMethod.jl/dev)

## Goal
Make `eigs` an efficient and native Julia function.

## Installation

Open the package manager in the REPL via `]` and run

```
(v1.6) pkg> add ArnoldiMethod
```

## Example

```julia
julia> using ArnoldiMethod, LinearAlgebra, SparseArrays

julia> A = spdiagm(
           -1 => fill(-1.0, 99),
            0 => fill(2.0, 100), 
            1 => fill(-1.0, 99)
       );

julia> decomp, history = partialschur(A, nev=10, tol=1e-6, which=SR());

julia> decomp
PartialSchur decomposition (Float64) of dimension 10
eigenvalues:
10-element Array{Complex{Float64},1}:
 0.0009674354160236865 + 0.0im
  0.003868805732811139 + 0.0im
  0.008701304061962657 + 0.0im
   0.01546025527344699 + 0.0im
  0.024139120518486677 + 0.0im
    0.0347295035554728 + 0.0im
   0.04722115887278571 + 0.0im
   0.06160200160067088 + 0.0im
    0.0778581192025522 + 0.0im
   0.09597378493453936 + 0.0im

julia> history
Converged: 10 of 10 eigenvalues in 174 matrix-vector products

julia> norm(A * decomp.Q - decomp.Q * decomp.R)
6.39386920955869e-8

julia> λs, X = partialeigen(decomp);

julia> norm(A * X - X * Diagonal(λs))
6.393869211477937e-8
```
