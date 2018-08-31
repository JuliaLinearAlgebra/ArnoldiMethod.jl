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
julia> decomp, history = partial_schur(A, nev=10, tol=1e-6, which=SR());
julia> history
Converged after 178 matrix-vector products
julia> norm(A * decomp.Q - decomp.Q * decomp.R)
3.717314639756976e-8
julia> λs, X = schur_to_eigen(decomp);
julia> norm(A * X - X * Diagonal(λs))
3.7173146389810755e-8
```