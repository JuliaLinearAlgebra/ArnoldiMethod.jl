# [Getting started](@id getting_started)

## Installing

In Julia open the package manager in the REPL via `]` and run:

```julia
(v1.0) pkg> add git@github.com:haampie/ArnoldiMethod.jl.git
```

Then use the package.

```julia
using ArnoldiMethod
```

## Construct a partial Schur decomposition

ArnoldiMethod.jl exports the `partialschur` function which can be used to 
obtain a partial Schur decomposition of any matrix `A`.

```@docs
partialschur
```

## From a Schur decomposition to an eigendecomposition
The eigenvalues and eigenvectors are obtained from the Schur form with the 
`partialeigen` function that is exported by ArnoldiMethod.jl:

```julia
λs, X = partialeigen(decomp::PartialSchur)
```

Note that whenever the matrix `A` is real-symmetric or Hermitian, the partial 
Schur decomposition coincides with the partial eigendecomposition, so in that 
case there is no need for the transformation.

## Example

Here we compute the first ten eigenvalues and eigenvectors of a tridiagonal
sparse matrix.

```julia
julia> using ArnoldiMethod, LinearAlgebra, SparseArrays
julia> A = spdiagm(
           -1 => fill(-1.0, 99),
            0 => fill(2.0, 100), 
            1 => fill(-1.0, 99)
       );
julia> decomp, history = partialschur(A, nev=10, tol=1e-6, which=SR());
julia> history
Converged after 178 matrix-vector products
julia> norm(A * decomp.Q - decomp.Q * decomp.R)
3.717314639756976e-8
julia> λs, X = partialeigen(decomp);
julia> norm(A * X - X * Diagonal(λs))
3.7173146389810755e-8
```
