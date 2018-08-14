using Test
using LinearAlgebra
using IRAM: partial_schur, schur_to_eigen

@testset "Schur to eigen" begin
    for T in (Float64, ComplexF64), i in 1:10
        A = rand(100,100)
        ε = 1e-6
        minim, maxim = 25, 35

        schur_decomp,  = partial_schur(A, min=minim, max=maxim, nev=minim, tol=eps(real(T)), maxiter=20)
        vals, vecs = schur_to_eigen(schur_decomp)

        @test norm(A*vecs - vals'.*vecs) < ε
    end
end