# ArnoldiMethod.jl

ArnoldiMethod.jl provides an iterative method to find a few approximate 
solutions to the eigenvalue problem in *standard form*:

```math
Ax = x\lambda,
```
where $A$ is a general matrix of size $n \times n$; and $x \in \mathbb{C}^n$ and
$\lambda \in \mathbb{C}$ are eigenvectors and eigenvalues respectively. By 
*general matrix* we mean that $A$ has no special structure. It can be symmetric
or non-symmetric and either real or complex.

The method is *matrix-free*, meaning that it only requires multiplication with 
the matrix $A$.

The package exports just two functions:
- [`partialschur`](@ref) to compute a stable basis for an eigenspace;
- [`partialeigen`](@ref) to compute an eigendecomposition from a partial Schur
  decomposition.

See **[Using ArnoldiMethod.jl](@ref getting_started)**  on how to use these 
functions.

## What algorithm is ArnoldiMethod.jl?

The underlying algorithm is the restarted Arnoldi method, which be viewed as a
mix between a subspace accelerated version of the power method and a truncated 
version of the dense QR algorithm.

Initially the method was based on the *Implicitly Restarted Arnoldi Method* (or
IRAM for short), which is the algorithm implemented by ARPACK. This method has a
very elegant restarting scheme based on exact QR iterations, but is 
unfortunately susceptible to forward instabilities of the QR algorithm.

For this reason the *Krylov-Schur* method is currently embraced in this package,
which is mathematically equivalent to IRAM, but has better stability by 
replacing exact QR iterations with a direct method that reorders the Schur form.
In fact we see Krylov-Schur just as an implementation detail of the Arnoldi 
method.

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
This project started with two goals:

- Having a *native* Julia implementation of the `eigs` function that performs as
  well as ARPACK. With native we mean that its implementation should be generic
  and support any number type. Currently the [`partialschur`](@ref) function 
  does not depend on LAPACK, and removing the last remnants of direct calls to 
  BLAS is in the pipeline.
- Removing the dependency of the Julia language on ARPACK. This goal was already
  achieved before the package was stable enough, since ARPACK moved to a 
  separate repository 
  [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl/).

## Status
An overview of what we have, how it's done and what we're missing.

### Implementation details

- The method does not make assumptions about the type of the matrix; it is 
  matrix-free.
- Converged Ritz vectors are locked (or deflated).
- Important matrices and vectors are pre-allocated and operations on the 
  Hessenberg matrix are in-place; Julia's garbage collector can sit back.
- Krylov basis vectors are orthogonalized with repeated classical Gram-Schmidt
  to ensure they are orthogonal up to machine precision; this is a BLAS-2
  operation.
- To compute the Schur decomposition of the Hessenberg matrix we use a dense 
  QR algorithm written natively in Julia. It is based on implicit (or Francis) 
  shifts and handles real arithmetic efficiently.
- Locking and purging of Ritz vectors is done by reordering the Schur form, 
  which is also implemented natively in Julia. In the real case it is done by
  casting tiny Sylvester equations to linear systems and solving them with 
  complete pivoting.
- Shrinking the size of the Krylov subspace and changing its basis is done by
  accumulating all rotations and reflections in a unitary matrix `Q`, and then
  simply computing the matrix-matrix product `V := V * Q`, where `V` is the 
  original orthonormal basis. This is not in-place in `V`, but with good reason: 
  the dense matrix-matrix product is not memory-bound.

### Not implemented (yet) and future ideas
- Being able to kickstart the method from a given Arnoldi relation. This also
  captures:
  1. Making an initial guess by providing a known approximate eigenvector;
  2. Deflating some subspace by starting the Arnoldi method with a given partial
     Schur decomposition.
- Matrix-induced inner product for generalized eigenvalue problems.
- Efficient implementation of symmetric problems with Lanczos.

On my wish list is to allow custom vector or matrix types, so that we can 
delegate expensive but trivial work to hardware that can do it faster 
(distributed memory / GPU). The basic concept would be: 

1. The core Arnoldi method performs tedious linear algebra on the projected, 
   low-dimensional problem, but finally just outputs a change of basis in the
   form of a unitary matrix Q.
2. Appropriate hardware does the change of basis `V := V * Q`.

Similar things should happen for expansion of the subspace and 
orthogonalization.