using ArnoldiMethod: Arnoldi, iterate_arnoldi!, reinitialize!, partialschur,
                     Reflector, reflector!, restore_hessenberg!, SR, LR
using SparseArrays
using Base: OneTo
using LinearAlgebra
using Random
using BenchmarkTools

function helloworld(n = 40, nev = 4)
    A = spdiagm(0 => [range(0.1, stop=5, length=n-3); 100:102])
    A[n,n-1] = 10.0
    A[n-1,n] = -10.0

    S, hist = partialschur(A, nev=nev, which = LR())

    @show norm(A * S.Q - S.Q * S.R)

    @show hist

    @show eigvals(S.R)
end
