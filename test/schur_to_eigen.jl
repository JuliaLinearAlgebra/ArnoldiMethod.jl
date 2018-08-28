using Test
using LinearAlgebra
using IRAM: partial_schur, schur_to_eigen
using Random

@testset "Schur to eigen $T take $i" for T in (Float64,ComplexF64), i in 1:10
    Random.seed!(i)
    A = randn(T, 100, 100)
    ε = 1e-7
    minim, maxim = 10, 20

    decomp, history = partial_schur(A, min=minim, max=maxim, nev=minim, tol=ε, restarts=100)
    @test history.converged

    vals, vecs = schur_to_eigen(decomp)

    # This test seems a bit flaky sometimes -- have to dig into it.
    for i = 1 : minim
        @test norm(A * vecs[:,i] - vecs[:,i] * vals[i]) < ε * abs(vals[i])
    end
end