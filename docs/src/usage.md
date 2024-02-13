# [Getting started](@id getting_started)

## Installing

In Julia open the package manager in the REPL via `]` and run:

```julia
(v1.6) pkg> add ArnoldiMethod
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

```@docs
partialeigen
```

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

## The PartialSchur and History structs

For completeness, the return values of the [`partialschur`](@ref) function:

```@docs
ArnoldiMethod.PartialSchur
ArnoldiMethod.History
```

## Bringing problems to standard form

ArnoldiMethod.jl by default can only compute an approximate, partial Schur decomposition
$AQ = QR$ and (from there) a partial eigendecomposition $AX = XD$ of a matrix $A$, for
*extremal* eigenvalues $d_{ii}$.

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

