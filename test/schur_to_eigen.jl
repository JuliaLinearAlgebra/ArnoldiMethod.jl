using Test
using LinearAlgebra, SparseArrays
using ArnoldiMethod: partialschur, partialeigen
using Random

@testset "Schur to eigen $T take $i" for T in (Float64,ComplexF64), i in 1:10
    Random.seed!(i)
    A = spdiagm(0 => 1:100) + sprand(100, 100, 0.01)
    ε = 1e-7
    minim, maxim = 10, 20

    decomp, history = partialschur(A, nev=minim, tol=ε, restarts=200)
    @test history.converged

    vals, vecs = partialeigen(decomp)

    # This test seems a bit flaky sometimes -- have to dig into it.
    for i = 1 : minim
        @test norm(A * vecs[:,i] - vecs[:,i] * vals[i]) < ε * abs(vals[i])
    end
end