# Spectral transformations

ArnoldiMethod.jl by default only solves the standard-form eigenvalue problem 
$Ax = x\lambda$ for $\lambda$ close to the boundary of the convex hull of 
eigenvalues.

Whenever one targets eigenvalues close to a specific point in the complex plane,
or whenever one solves generalized eigenvalue problems, spectral transformations
will enable you to recast the problem into something that ArnoldiMethod.jl can 
handle well. In this section we provide some examples.

## Shift-and-invert with LinearMaps.jl
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
decomp, = partialschur(construct_linear_map(A), nev=4, tol=1e-5, restarts=100, which=LM())
λs_inv, X = partialeigen(decomp)

# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
λs = 1 ./ λs_inv
 
# Show that Ax = xλ
@show norm(A * X - X * Diagonal(λs)) # 7.38473677258669e-6
```

# [Smallest eigenvalues of generalized eigenvalue problem](@id generalized_shift_invert)
When targeting the eigenvalues closest to the origin of a generalized eigenvalue
problem $Ax = Bx\lambda$, one can apply the shift-and-invert trick, recasting 
the problem to $A^{-1}Bx = x\theta$ where $\lambda = 1 / \theta$.

```julia
using ArnoldiMethod, LinearAlgebra, LinearMaps

# Define the matrices of the generalized eigenvalue problem
A, B = rand(100,100), rand(100,100)

struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y,x)
    mul!(M.temp, M.B, x)
    ldiv!(y, M.A_lu, M.temp)
end

function construct_linear_map(A,B)
    a = ShiftAndInvert(factorize(A),B,Vector{eltype(A)}(undef, size(A,1)))
    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
end

# Target the largest eigenvalues of the inverted problem
decomp,  = partialschur(construct_linear_map(A, B), nev=4, tol=1e-5, restarts=100, which=LM())
λs_inv, X = partialeigen(decomp)

# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
λs = 1 ./ λs_inv

# Show that Ax = Bxλ
@show norm(A * X - B * X * Diagonal(λs)) # 2.8043149027575927e-6
```