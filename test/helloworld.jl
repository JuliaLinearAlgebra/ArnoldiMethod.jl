using IRAM
using SparseArrays, LinearAlgebra
using Plots

function helloworld()
    A = randn(500, 500)
    @time λs = eigvals(A)
    @time schur, prods = partial_schur(A, min=10, max=30, tol=1e-6, maxiter=100, which=SR())
    θs = eigvals(schur.R)

    p = scatter(real(λs), imag(λs), label = "All eigenvalues", aspect_ratio = :equal)
    scatter!(real(θs), imag(θs), mark = :+, label = "IRAM")

    @show norm(A * schur.Q - schur.Q * schur.R) prods

    return A, schur, p
end