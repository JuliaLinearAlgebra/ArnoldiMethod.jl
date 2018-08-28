using ArnoldiMethod
using SparseArrays, LinearAlgebra
using Plots

function helloworld()
    A = randn(500, 500)
    @time λs = eigvals(A)
    @time schur, history = partial_schur(A, min=15, max=30, tol=1e-6, restarts=100, which=SR())

    if history.converged
        @warn "Not converged :("
    end

    θs = eigvals(schur.R)
    p = scatter(real(λs), imag(λs), label = "All eigenvalues", aspect_ratio = :equal)
    scatter!(real(θs), imag(θs), mark = :+, label = "IRAM")
    @show norm(A * schur.Q - schur.Q * schur.R) history

    return A, schur, p
end