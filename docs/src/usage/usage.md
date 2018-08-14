# Using IRAM.jl

You can compute the partial Schur form of a matrix with `partial_schur`:

`partial_schur(A; min, max, nev, tol, maxiter, which)`

where `A` is the `n`×`n` matrix whose eigenvalues and eigenvectors you want; `min` specifies the minimum dimension to which the Hessenberg matrix is reduced after `implicit_restart!`; `max` specifies the maximum dimension of the Hessenberg matrix at which point `iterate_arnoldi!` stops increasing the dimension of the Krylov subspace; `nev` specifies the minimum amount of eigenvalues the method gives you; `tol` specifies the criterion that determines when the eigenpairs are considered converged (in practice, smaller `tol` forces the eigenpairs to converge even more); `maxiter` specifies the maximum amount of restarts the method can perform before ending the iteration; and `which` is a `Target` structure and specifies which eigenvalues are desired (Largest magnitude, smallest real part, etc.).

The function call `partial_schur` returns a tuple (`P`,`prods`), where `P` is a `PartialSchur` struct with fields `Q` and `R` for which `A * P.Q ≈ P.Q * P.R`. The upper triangular matrix `R` is of size `nev`×`nev` and the unitary matrix `Q` is of size `n`×`nev`. The amount of matrix-vector products computed during the iterations is stored in `prods`.

You can compute the eigenvalues and eigenvectors from the Schur form with `schur_to_eigen`:

`schur_to_eigen(P)`

where `P` is a `PartialSchur` struct that contains the partial Schur decomposition of a matrix `A`. This computes the eigenvalues and eigenvectors of matrix `A` from the Schur decomposition `P`.

An example of how to use IRAM.jl's function `partial_schur`:
```
julia> using IRAM, LinearAlgebra
# Generate a sparse matrix
julia> A = spdiagm(-1 => fill(-1.0, 99), 0 => fill(2.0, 100), 1 => fill(-1.001, 99));
# Compute Schur form of A
julia> schur_form,  = partial_schur(A, min = 12, max = 30, nev = 10, tol = 1e-10, maxiter = 20, which=LM());
julia> Q,R = schur_form.Q, schur_form.R;
julia> norm(A*Q - Q*R)
6.336794280593682e-11
# Compute eigenvalues and eigenvectors of A
julia> vals, vecs = schur_to_eigen(schur_form);
# Show that Ax = λx
julia> norm(A*vecs - vecs*Diagonal(vals))
6.335460143979987e-11
```

# Applying shift-and-invert to target smallest eigenvalues with LinearMaps.jl
```
using IRAM, LinearMaps

# Define a matrix whose eigenvalues you want
A = rand(100,100)

# Inverts the problem
function construct_linear_map(A)
    a = factorize(A)
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, a, x), size(A,1), ismutating=true)
end

# Target the largest eigenvalues of the inverted problem
schur_form,  = partial_schur(construct_linear_map(A), min=11, max=22, nev=10, tol=1e-5, maxiter=100, which=LM())
inv_vals, vecs = schur_to_eigen(schur_form)

# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
vals = ones(length(inv_vals))./inv_vals

# Show that Ax = λx
@assert norm(A*vecs - vecs*Diagonal(vals)) < 1e-5
```

# Solving a generalized eigenvalue problem targeting smallest eigenvalues with LinearMaps.jl
```
using IRAM, LinearMaps

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
schur_form,  = partial_schur(construct_linear_map(A,B), min=11, max=22, nev=10, tol=1e-5, maxiter=100, which=LM())
inv_vals, vecs = schur_to_eigen(schur_form)

# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
vals = ones(length(inv_vals))./inv_vals

# Show that Ax = λBx
@assert norm(A*vecs - B*vecs*Diagonal(vals)) < 1e-5
```