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

## Installing

In Julia open the package manager in the REPL via `]` and run:

```julia
(v1.6) pkg> add ArnoldiMethod
```

Then use the package.

```julia
using ArnoldiMethod
```

The package exports just two functions:
- [`partialschur`](@ref) to compute a stable basis for an eigenspace;
- [`partialeigen`](@ref) to compute an eigendecomposition from a partial Schur
  decomposition.

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


## Partial Schur decomposition

The [`partialschur`](@ref) method forms the backbone of the package. It computes
an approximate, partial Schur decomposition of a matrix $A$:

```math
AQ = QR
```

where $Q$ is orthonormal of size $n \times \texttt{nev}$ and $R$ is upper 
triangular of size $\texttt{nev} \times \texttt{nev}.$ with eigenvalues of
$A$ on the diagonal.

!!! note "2x2 blocks in real arithmetic"

    In real arithmetic $R$ is quasi upper triangular, with $2 \times 2$ blocks on the
    diagonal  corresponding to conjugate complex-valued eigenpairs. These $2 \times 2$
    blocks are used for efficiency, since otherwise the entire Schur form would have to
    use complex arithmetic.

!!! note "A partial Schur decomposition is often enough"

    Often a partial Schur decomposition is all you need, cause it's more stable
    to compute and work with than a partial eigendecomposition.
    
    Also note that for complex Hermitian and real symmetric matrices, the partial
    Schur form coincides with the partial eigendecomposition (the $R$ matrix is
    diagonal).

```@docs
partialschur
partialschur!
```

## Partial eigendecomposition

After computing the partial Schur decomposition, it can be transformed into a
partial eigendecomposition via the [`partialeigen`](@ref) helper function. The
basic math is to determine the eigendecomposition of the upper triangular matrix
$RY = Y\Lambda$ such that

```math
A(QY) = (QY)\Lambda
```

forms the full eigendecomposition of $A$, where $QY$ are the eigenvectors and
$\Lambda$ is a $\texttt{nev} \times \texttt{nev}$ diagonal matrix of eigenvalues.

This step is a relatively cheap post-processing step.

```@docs
partialeigen
```

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


## The PartialSchur and History structs

For completeness, the return values of the [`partialschur`](@ref) function:

```@docs
ArnoldiMethod.PartialSchur
ArnoldiMethod.History
```

## Passing an initial guess

If you have a good guess for a target eigenvector, you can potentially speed up
the method by passing it through `partialschur(A, v1=my_initial_vector)`. This
vector is then used to build the Krylov subspace.

## Pre-allocating and custom matrix types

If you call `partialschur` multiple times, and you want to allocate large arrays and buffers only
once ahead of time, you can allocate the relevant matrices manually and pass them to the algorithm.

The same can be done if you want to work with custom matrix types.

```@docs
ArnoldiMethod.ArnoldiWorkspace
```

## Starting from an initial partial Schur decomposition

You can also use [`ArnoldiWorkspace`](@ref) to start the algorithm from an initial partial Schur
decomposition. This is useful if you already found a few Schur vectors, and want to continue to
find more.

```julia
A = rand(100, 100)

# Pre-allocate the relevant Krylov subspace arrays
V, H = rand(100, 21), rand(21, 20)
arnoldi = ArnoldiWorkspace(V, H)

# Find a few eigenvalues
_, info_1 = partialschur!(A, arnoldi, nev = 3, tol = 1e-12)

# Then continue to find a couple more. Notice: 5 in total, so 2 more. Allow larger errors by
# changin `tol`.
F, info_2 = partialschur!(A, arnoldi, nev = 5, start_from = info_1.nconverged + 1 , tol = 1e-8)
@show norm(A * F.Q - F.Q * F.R)
```

!!! note "Setting `start_from` correctly"

    As pointed out above, in real arithmetic the algorithm may find one eigenvalue more than
    requested if it corresponds to a conjugate pair. Also it may find fewer, if not all
    converge. So if you reuse your `ArnoldiWorkspace`, make sure to set `start_from` to one plus
    the number of previously converged eigenvalues.

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


## Bringing problems to standard form

ArnoldiMethod.jl by default can only compute an approximate, partial Schur decomposition
$AQ = QR$ and (from there) a partial eigendecomposition $AX = XD$ of a matrix $A$, for
*extremal* eigenvalues $d_{ii}$. These are eigenvalues at the boundary of the convex
hull of the spectrum of $A$. Search targets for eigenvalues are: large magnitude, and
large or small real or imaginary parts.

Whenever one targets eigenvalues close to a specific point in the complex plane,
or whenever one solves generalized eigenvalue problems, suitable transformations
will enable you to recast the problem into something that ArnoldiMethod.jl can 
handle well. In this section we provide some examples.

### Shift-and-invert with LinearMaps.jl
To find eigenvalues closest to the origin of $A$, one can find the eigenvalues
of largest magnitude of $A^{-1}$. [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl) 
is a neat way to implement this.

```julia
using ArnoldiMethod, LinearAlgebra, LinearMaps

# Define a matrix whose eigenvalues you want
A = rand(100,100)

# Factorizes A and builds a linear map that applies inv(A) to a vector.
function construct_linear_map(A)
    F = factorize(A)
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A,1), ismutating=true)
end

# Target the largest eigenvalues of the inverted problem
decomp, = partialschur(construct_linear_map(A), nev=4, tol=1e-5, restarts=100, which=:LM)
λs_inv, X = partialeigen(decomp)

# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
λs = 1 ./ λs_inv
 
# Show that Ax = xλ
@show norm(A * X - X * Diagonal(λs)) # 7.38473677258669e-6
```

### Shift-and-invert for generalized eigenvalue problems
When targeting the eigenvalues closest to the origin of a generalized eigenvalue
problem $Ax = Bx\lambda$, one can apply the shift-and-invert trick, recasting 
the problem to $A^{-1}Bx = x\theta$ where $\lambda = 1 / \theta$.

```julia
using ArnoldiMethod, LinearAlgebra, LinearMaps

# Define the matrices of the generalized eigenvalue problem
A, B = rand(100, 100), rand(100, 100)

struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y, x)
    mul!(M.temp, M.B, x)
    ldiv!(y, M.A_lu, M.temp)
end

function construct_linear_map(A, B)
    a = ShiftAndInvert(factorize(A), B, Vector{eltype(A)}(undef, size(A, 1)))
    LinearMap{eltype(A)}(a, size(A, 1), ismutating = true)
end

# Target the largest eigenvalues of the inverted problem
decomp, = partialschur(
    construct_linear_map(A, B),
    nev = 4,
    tol = 1e-5,
    restarts = 100,
    which = :LM,
)
λs_inv, X = partialeigen(decomp)

# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
λs = 1 ./ λs_inv

# Show that Ax = Bxλ
@show norm(A * X - B * X * Diagonal(λs)) # 2.8043149027575927e-6
```

### Largest eigenvalues of a generalized eigenvalue problem with symmetric positive-definite B

When $B$ is a symmetric positive-definite matrix, and it's affordable to compute a Cholesky
decomposition of $B$, one can use ArnoldiMethod.jl to create a partial Schur decomposition of
$A$ where the Schur vectors are $B$-orthonormal:

Solve $Q^*AQ = R$ where $Q^*BQ = I$ and $R$ is upper triangular. If $A = A^*$ as well, then
$R$ is diagonal and we have a partial eigendecomposition of $A$.

First we take the Cholesky decomposition $B = LL^*$ and plug it into $AQ = BQR$ to obtain
$L^{-*}AL^{-1}L^*Q = L^*QR$.

Then define $C = L^{-*}AL^{-1}$ and $Y = L^*Q$ and we have a standard Schur decomposition
$CY = YR$ which we can solve using `partialschur`.

The linear map $C$ can be defined as follows:

```julia
using ArnoldiMethod, LinearAlgebra, LinearMaps
struct WithBInnerProduct{TA,TL}
    A::TA
    L::TL
end

function (M::WithBInnerProduct)(y, x)
    # Julia unfortunately does not have in-place CHOLMOD solve, so this is far from optimal.
    tmp = M.L \ x
    mul!(y, M.A, tmp)
    y .= M.L' \ y
    return y
end

# Define the matrices of the generalized eigenvalue problem
A = rand(100, 100)
B = Diagonal(range(1.0, 2.0, length = 100))

# Reformulate the problem as standard Schur decomposition
F = cholesky(B)
linear_map = LinearMap{eltype(A)}(WithBInnerProduct(A, F.L), size(A, 1), ismutating = true)
decomp, info = partialschur(linear_map, nev = 4, which = :LM, tol = 1e-10)

# Translate back to the original problem
Q = F.L' \ decomp.Q

@show norm(Q' * A * Q - decomp.R)  # 3.883933945390996e-14
@show norm(Q' * B * Q - I)  # 3.1672155003480583e-15
```


## Goal of this package: an efficient, pure Julia implementation
This project started with two goals:

1. Having a *native* Julia implementation of the `eigs` function that performs as
   well as ARPACK. With native we mean that its implementation should be generic
   and support any number type. Currently the [`partialschur`](@ref) function
   does not depend on LAPACK, it even has its own implementation of a dense
   eigensolver.
2. Removing the dependency of the Julia language on ARPACK. This goal was already
   achieved before the package was stable enough, since ARPACK moved to a
   separate repository [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl/).


## Performance

ArnoldiMethod.jl should be roughly on par with Arpack.jl, and slightly faster than
KrylovKit.jl.

Do note that for an apples to apples comparison, it's important to compare with
identical defaults: each of the mentioned packages uses a slightly different default
convergence criterion.


## Status
An overview of what we have, how it's done and what we're missing.

### Implementation details

- The method does not make assumptions about the type of the matrix; it is 
  matrix-free.
- Converged Ritz vectors are locked (or deflated).
- We may do "purging" differently from ARPACK: in ArnoldiMethod.jl it is rather
  "unlocking", in the sense that converged but unwanted eigenvectors are retained
  in the search subspace instead of removed from it.
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
  original orthonormal basis. This is not in-place in `V`, so we allocate a bit
  of scratch space ahead of time.
