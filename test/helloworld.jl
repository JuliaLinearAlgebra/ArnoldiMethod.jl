using IRAM
using SparseArrays, LinearAlgebra
using Plots

function helloworld()
    A = randn(1000, 1000)
    @time λs = eigvals(A)
    @time schur, = partial_schur(A, min=10, max=30, tol=1e-3, maxiter=100)
    θs = eigvals(schur.R)

    p = scatter(real(λs), imag(λs), label = "All eigenvalues", aspect_ratio = :equal)
    scatter!(real(θs), imag(θs), mark = :+, label = "IRAM")

    return p
end