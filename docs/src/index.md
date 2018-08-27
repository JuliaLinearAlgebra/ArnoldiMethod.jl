# IRAM.jl

IRAM.jl provides an iterative method to find a few approximate solutions to the 
eigenvalue problem in *standard form*:

```math
Ax = x\lambda,
```
where $A$ is a general matrix of size $n \times n$; and $x \in \mathbb{C}^n$ and
$\lambda \in \mathbb{C}$ are eigenvectors and eigenvalues respectively. By 
*general matrix* we mean that $A$ has no special structure. It can be symmetric
or non-symmetric and either real or complex.

The method is *matrix-free*, meaning that it only requires multiplication with 
the matrix $A$.

See **[Using IRAM.jl](@ref getting_started)** on how to use the package.

## What algorithm is IRAM.jl?

The underlying algorithm is the Implicitly Restarted Arnoldi Method, which be 
viewed as a mix between a subspace accelerated version of the power method and 
a truncated version of the dense QR algorithm.

## What problems does this package solve specifically?

By design the Arnoldi method is best at finding eigenvalues on the boundary of
the convex hull of eigenvalues. For instance eigenvalues of largest modulus and
largest or smallest real part. In the case of complex matrices one can target
eigenvalues of largest and smallest imaginary part as well.

The scope is much broader though, since there is a whole zoo of spectral 
transformations possible to find for instance interior eigenvalues or 
eigenvalues closest to the imaginary axis.

Further, one can solve generalized eigenvalue problems $Ax = Bx \lambda$ by
applying a suitable spectral transformation as well. Also quadratic eigenvalue 
problems can be casted to standard form.

See [Theory](@ref theory) for more information.

## Goal of this package: a pure Julia implementation
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
Currently features:

- An efficient dense QR algorithm natively in Julia, used to do implicit
  restarts and to compute the low-dimensional dense eigenproblem involving the
  Hessenberg matrix. It is based on implicit shifts and handles real arithmetic
  efficiently;
- Transforming converged Ritz values to a partial Schur form.

Work in progress:
- Native transformation of real Schur vectors to eigenvectors.
- Using a matrix-induced inner product in the case of generalized eigenvalue
  problems.
- Efficient implementation of symmetric problems with Lanczos.