# ArnoldiMethod.jl

[![Build Status](https://travis-ci.org/haampie/ArnoldiMethod.jl.svg?branch=master)](https://travis-ci.org/haampie/ArnoldiMethod.jl) [![codecov](https://codecov.io/gh/haampie/ArnoldiMethod.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/haampie/ArnoldiMethod.jl)


The Implicitly Restarted Arnoldi Method, natively in Julia.

## Docs
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://haampie.github.io/ArnoldiMethod.jl/stable) [![Latest docs](https://img.shields.io/badge/docs-latest-gray.svg)](https://haampie.github.io/ArnoldiMethod.jl/latest)

## Goal
Make `eigs` a native Julia function.

## Installation

Open the package manager in the REPL via `]` and run

```
(v1.0) pkg> add ArnoldiMethod
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
 0.000967435416023798 + 0.0im
 0.003868805732811847 + 0.0im
 0.008701304061962362 + 0.0im
 0.015460255273447325 + 0.0im
  0.02413912051848671 + 0.0im
   0.0347295035555462 + 0.0im
 0.047221158872786585 + 0.0im
 0.061602001600669004 + 0.0im
  0.07785811920255274 + 0.0im
    3.918903416103043 + 0.0im
julia> history
Converged: 10 of 10 eigenvalues in 171 matrix-vector products
julia> norm(A * decomp.Q - decomp.Q * decomp.R)
2.503582041203943e-7
julia> λs, X = partialeigen(decomp);
julia> norm(A * X - X * Diagonal(λs))
2.503582040967213e-7
```