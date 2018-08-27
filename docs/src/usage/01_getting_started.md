# [Getting started](@id getting_started)

## Installing

In Julia open the package manager in the REPL via `]` and run:

```julia
(v1.0) pkg> add git@github.com:haampie/IRAM.jl.git
```

Then use the package.

```julia
using IRAM
```

## Construct a partial Schur decomposition

IRAM.jl exports the `partial_schur` function which can be used to obtain a 
partial Schur decomposition of any matrix `A`:

```julia
decomp, prods, converged = partial_schur(A; nev::Int, tol::Real, which::Target, min, max, maxiter)
```

From a user perspective, the interesting parameters are `nev`, `tol` and `which`:

| Argument |  Description                                                                                                                              |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------|
| nev      | Number of eigenvalues                                                                                                                     |
| tol      | Tolerance for the stopping criterion                                                                                                      |
| which    | Which eigenvalues do you want? In real arithmetic, choose **largest magnitude** `LM()`, **largest real part** `LR()`, or **smallest real part** `SR()`. In complex arithmetic additionally **largest imaginary part** `LI()` and **smallest imaginary part** `SI()` |

The other keyword arguments have sensible values by default and are for advanced
use. When the algorithm does not converge, one can increase `maxiter`. When the
algorithm converges to slowly, one can play with `min` and `max`. It is 
suggested to keep `min` equal to or slighly larger than `nev`, and `max` is 
usually about two times `min`.

The function returns a tuple with values:

| Return value | Description                                                                                                                         |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------|
| decomp       | A `PartialSchur` object with orthonormal matrix `decomp.Q` and (quasi) upper triangular `decomp.R.` for which `A * P.Q ≈ P.Q * P.R` |
| prods        | Number of matrix-vector products that were necessary                                                                                |
| converged    | Boolean indicating whether all eigenvalues have converged                                                                           |

## From a Schur decomposition to an eigendecomposition
The eigenvalues and eigenvectors are obtained from the Schur form with the 
`schur_to_eigen` function that is exported by IRAM.jl:

```julia
λs, X = schur_to_eigen(decomp::PartialSchur)
```

## Example

Here we compute the first ten eigenvalues and eigenvectors of a tridiagonal
sparse matrix.

```julia
julia> using IRAM, LinearAlgebra, SparseArrays
julia> A = spdiagm(
           -1 => fill(-1.0, 99),
            0 => fill(2.0, 100), 
            1 => fill(-1.0, 99)
       );
julia> decomp, prods, converged = partial_schur(A, nev=10, max=30, tol=1e-6, which=SR(), maxiter=100);
julia> converged
true
julia> norm(A * decomp.Q - decomp.Q * decomp.R)
6.723110275944552e-9
julia> λs, X = schur_to_eigen(decomp);
julia> norm(A * X - X * Diagonal(λs))
6.7231102983472125e-9
```
