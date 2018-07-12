# IRAM.jl

IRAM.jl approximately solves the eigenproblem `Ax = λx` where `A` is a large, sparse and non-symmetric matrix. It is a matrix-free method, and only requires multiplications with 
`A`. It is based on the implicitly restarted Arnoldi method, which be viewed as a mix between
a subspace accelerated version of the power method and a truncated version of the dense QR
algorithm.

## Pure Julia implementation
The algorithm is a pure Julia implementation of the implicitly restarted Arnoldi method and
is loosely based on ARPACK. It is not our goal to make an exact copy of ARPACK. With "pure
Julia" we mean that we do not rely on LAPACK for linear algebra routines. This allows us to
use any number type.

When this project started, ARPACK was still a dependency of the Julia language, and the main
goal was to get rid of this. Currently ARPACK has moved to a separate repository called 
[Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl/), but still it would be great
to have a native Julia implementation of this algorithm.

## Status
Still a work in progress!
- [x] Efficient QR iterations in implicit restart;
- [x] Real arithmetic with real matrices by handling conjugate eigenpairs efficiently;
- [ ] Targeting of eigenvalues;
- [x] Locking of converged Ritz vectors;
- [ ] Generalized eigenvalue problems.


