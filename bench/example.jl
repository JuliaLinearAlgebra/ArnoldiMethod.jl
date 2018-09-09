using ArnoldiMethod
using SparseArrays
using Base: OneTo
using LinearAlgebra
using Random
using BenchmarkTools
using Plots
using Arpack

function helloworld(n = 40, nev = 4)
    A = randn(n, n)
    @time S, hist = partialschur(A, nev=nev, which = LR(), tol = 1e-6)
    @time eigs(A, nev=nev, which = :LR, tol = 1e-6)
    @show norm(A * S.Q - S.Q * S.R)
    println(hist)
    θs = eigvals(S.R)
    @time λs = eigvals(A)

    scatter(real(λs), imag(λs), aspect_ratio = :equal)
    scatter!(real(θs), imag(θs), mark = :+)
end
