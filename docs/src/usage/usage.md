# Using IRAM.jl

An example of how to use IRAM.jl's function `partial_schur`:
```
julia> using IRAM
julia> A = rand(100, 100)
julia> schur_form = partial_schur(A, min = 12, max = 30, nev = 10, tol = 1e-10, maxiter = 20, which=LM())
julia> Q,R = schur_form.Q, schur_form.R
julia> norm(A*Q - Q*R)
1.4782234971282938e-14
```