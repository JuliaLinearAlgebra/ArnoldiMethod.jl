# Eigenvalue problems and transformations

In this section we explore several ways to use IRAM.jl to solve the generalized
eigenvalue problem

```math
Ax = \lambda Bx.
```
This problem arises for instance in:

1. Finite element discretizations, with $B$ a symmetric, positive definite mass 
   matrix;
2. Stability analysis of Navier-Stokes equations, where $B$ is semi-definite 
   and singular;
3. Simple finite differences discretizations where typically $B = I.$

Because IRAM.jl only deals with the standard form

```math
Cx = \lambda x.
```
we have to do a spectral transformation whenever $B \neq I.$ Secondly, to get
fast convergence, one typically applies shift-and-invert techniques, which also
requires a spectral transformation.

## Transformation to standard form for non-singular B
If $B$ is nonsingular and easy to factorize, one can define the matrix $C = B^{-1}A$
and apply IRAM to the eigenproblem

```math
Cx = \lambda x
```

which is in standard form. Of course $C$ should not be formed explicity! One only
has to provide the action of the matrix-vector product by implementing
`LinearAlgebra.mul!(y, C, x)`. The best way to do so is to factorize $B$ up front.

IRAM.jl does not yet provide helper functions for this transformation.


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

IRAM.jl does not transform the eigenvalues of $C$ back to the eigenvalues of
$(A, B).$ However, the relation is simply $\lambda = \sigma + \theta^{-1}$.

## Purification

If $B$ is exactly singular or very ill-conditioned, one cannot work with 
$C = B^{-1}A$. One can however apply the shift-and-invert method. There is a 
catch, since $C = (A - \sigma B)^{-1}B$ has eigenvalues close to zero
or exactly zero. When transformed back, these values would corrspond to 
$\lambda = \infty.$ The process to remove these eigenvalues is called 
purification.

IRAM.jl does not yet support this purification idea, but it could in principle
be put together along the following lines [^MLA]:
1. Start the Arnoldi method with $C$ times a random vector, such that the 
   initial vector has numerically no components in the null space of $C$.
2. Expand the Krylov subspace with one additional vector and add a zero shift.
   This corresponds to implicitly multiplying the first vector of the Krylov
   subspace with $C$ at restart.

[^MLA]: Meerbergen, Karl, and Alastair Spence. "Implicitly restarted Arnoldi with purification for the shift-invert transformation." Mathematics of Computation of the American Mathematical Society 66.218 (1997): 667-689.