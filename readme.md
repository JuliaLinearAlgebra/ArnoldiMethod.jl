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

julia> decomp, history = partialschur(A, nev=10, tol=1e-6, which=:SR);

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

## ArnoldiMethod.jl is generic

ArnoldiMethod.jl's Schur decomposition is written in Julia, it does not use LAPACK. This allows
you to use arbitrary number types.

We repeat the above example with [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)
and more accuracy.


```julia
julia> using ArnoldiMethod, DoubleFloats, LinearAlgebra, SparseArrays

julia> A = spdiagm(
           -1 => fill(Double64(-1), 99),
            0 => fill(Double64(2), 100), 
            1 => fill(Double64(-1), 99)
       );

julia> decomp, history = partialschur(A, nev=10, tol=1e-28, which=:SR);

julia> decomp
PartialSchur decomposition (Double64) of dimension 10
eigenvalues:
10-element Vector{Complex{Double64}}:
  9.6743541602387015850892187143202406e-04 + 0.0im
 3.86880573281130335530623278634505297e-03 + 0.0im
 8.70130406196283903200426213162702754e-03 + 0.0im
 1.54602552734469798152574737604660783e-02 + 0.0im
 2.41391205184865585041130463401142985e-02 + 0.0im
 3.47295035554726251259365854776375027e-02 + 0.0im
 4.72211588727859409278578405476287512e-02 + 0.0im
 6.16020016006677741124091774018629622e-02 + 0.0im
 7.78581192025509024705094505968950069e-02 + 0.0im
 9.59737849345402152393882633733121172e-02 + 0.0im

julia> history
Converged: 10 of 10 eigenvalues in 442 matrix-vector products

julia> norm(A * decomp.Q - decomp.Q * decomp.R)
4.53243232681764960018699535610331068e-30

julia> norm(decomp.Q' * decomp.Q - I)
3.53573060252329801278244497021683397e-29
```