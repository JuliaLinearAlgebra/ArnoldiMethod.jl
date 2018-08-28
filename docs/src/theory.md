# [Standard-form eigenvalue problems](@id theory)

ArnoldiMethod.jl is intended to find a few approximate solutions to the 
eigenvalue problem

```math
Ax = x \lambda
```

This problem is handled in two steps:

1. For numerical stability, the method firstly constructs a partial Schur form
   ```math
   AQ = QR
   ```
   where $Q$ is orthonormal of size $n \times \texttt{nev}$ and $R$ is upper 
   triangular of size $\texttt{nev} \times \texttt{nev}.$ In real arithmetic $R$
   is quasi upper triangular, with $2 \times 2$ blocks on the diagonal 
   corresponding to conjugate complex-valued eigenpairs.
2. The user can transform the partial Schur form into an eigendecomposition via
   a helper function. The basic math is to determine the eigendecomposition of
   the upper triangular matrix $RY = Y\Lambda$ such that
   ```math
   A(QY) = (QY)\Lambda
   ```
   forms the full eigendecomposition.

Step 2 is a cheap post-processing step. Also note that it is not necessary when 
the matrix is symmetric, because in that case the Schur decomposition coincides 
with the eigendecomposition.

## Stopping criterion
ArnoldiMethod.jl considers an approximate eigenpair converged when the 
condition

```math
\|Ax - x\lambda\|_2 < \texttt{tol}|\lambda|
```

is met, where `tol` is a user provided tolerance. Note that this stopping 
criterion is scale-invariant. For a scaled matrix $B = \alpha A$ the same 
approximate eigenvector together with the scaled eigenvalue $\alpha\lambda$ 
would satisfy the stopping criterion.

## Spectral transformations

There are multiple reasons to use a spectral transformation. Firstly, consider
the generalized eigenvalue problem

```math
Ax = Bx\lambda.
```

This problem arises for instance in:

1. Finite element discretizations, with $B$ a symmetric, positive definite mass 
   matrix;
2. Stability analysis of Navier-Stokes equations, where $B$ is semi-definite 
   and singular;
3. Simple finite differences discretizations where typically $B = I.$

Because ArnoldiMethod.jl only deals with the standard form

```math
Cx = x\lambda.
```
we have to do a spectral transformation whenever $B \neq I.$ 

Secondly, to get fast convergence, one typically applies shift-and-invert 
techniques, which also requires a spectral transformation.

## Transformation to standard form for non-singular B
If $B$ is nonsingular and easy to factorize, one can define the matrix 
$C = B^{-1}A$ and apply the Arnoldi method to the eigenproblem

```math
Cx = x\lambda
```

which is in standard form. Of course $C$ should not be formed explicity! One only
has to provide the action of the matrix-vector product by implementing
`LinearAlgebra.mul!(y, C, x)`. The best way to do so is to factorize $B$ up front.

See [an example here](@ref generalized_shift_invert).

## Targeting eigenvalues with shift-and-invert
When looking for eigenvalues near a specific target $\sigma$, one can get fast 
convergence by using a shift-and-invert technique:

```math
Ax = \lambda Bx. \iff (A - \sigma B)^{-1}Bx = \theta x \text{ where } \theta = \frac{1}{\lambda - \sigma}.
```

Here we have casted the generalized eigenvalue problem into standard form with a
matrix $C = (A - \sigma B)^{-1}B.$ Note that $\theta$ is large whenever $\sigma$
is close to $\lambda$. This means that we have to target the **largest** 
eigenvalues of $C$ in absolute magnitude. Fast convergence is guaranteed 
whenever $\sigma$ is close enough to an eigenvalue $\lambda$.

Again, one should not construct the matrix $C$ explicitly, but rather implement
the matrix-vector product `LinearAlgebra.mul!(y, C, x)`. The best way to do so
is to factorize $A - \sigma B$ up front.

Note that this shift-and-invert strategy simplifies when $B = I,$ in which case
the matrix in standard form is just $C = (A - \sigma I)^{-1}.$

ArnoldiMethod.jl does not transform the eigenvalues of $C$ back to the 
eigenvalues of $(A, B).$ However, the relation is simply 
$\lambda = \sigma + \theta^{-1}$.

## Purification

If $B$ is exactly singular or very ill-conditioned, one cannot work with 
$C = B^{-1}A$. One can however apply the shift-and-invert method. There is a 
catch, since $C = (A - \sigma B)^{-1}B$ has eigenvalues close to zero
or exactly zero. When transformed back, these values would corrspond to 
$\lambda = \infty.$ The process to remove these eigenvalues is called 
purification.

ArnoldiMethod.jl does not yet support this purification idea, but it could in 
principle be put together along the following lines [^MLA]:
1. Start the Arnoldi method with $C$ times a random vector, such that the 
   initial vector has numerically no components in the null space of $C$.
2. Expand the Krylov subspace with one additional vector and add a zero shift.
   This corresponds to implicitly multiplying the first vector of the Krylov
   subspace with $C$ at restart.

[^MLA]: Meerbergen, Karl, and Alastair Spence. "Implicitly restarted Arnoldi with purification for the shift-invert transformation." Mathematics of Computation of the American Mathematical Society 66.218 (1997): 667-689.