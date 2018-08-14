# IRAM.jl

IRAM.jl finds multiple approximate solutions to the eigenproblem 
$Ax = \lambda x$ where $A$ is a large, sparse and non-symmetric matrix. 
It is a matrix-free method, and only requires multiplications with $A$. 
It is based on the implicitly restarted Arnoldi method, which be viewed as a 
mix between a subspace accelerated version of the power method and a truncated 
version of the dense QR algorithm.

Via spectral transformations one can use this package to solve generalized
eigenvalue problems $Ax = \lambda Bx,$ see 
[Eigenvalue problems and transformations](@ref).

## Pure Julia implementation
The algorithm is a pure Julia implementation of the implicitly restarted 
Arnoldi method and is loosely based on ARPACK. It is not our goal to make an 
exact copy of ARPACK. With "pure Julia" we mean that we do not rely on LAPACK 
for linear algebra routines. This allows us to use any number type. In some 
occasions we do rely on BLAS.

When this project started, ARPACK was still a dependency of the Julia language, 
and the main goal was to get rid of this. Currently ARPACK has moved to a 
separate repository called 
[Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl/), but still it 
would be great to have a native Julia implementation of this algorithm.

## Status
Still a work in progress! Currently we have:

- An efficient dense QR algorithm natively in Julia, used to do implicit
  restarts and to compute the low-dimensional dense eigenproblem involving the
  Hessenberg matrix. It is based on implicit shifts and handles real arithmetic
  efficiently;
- Locking of converged Ritz vectors

Work in progress:
- Efficient transformation of real Schur vectors to eigenvectors.
- Search targets.



