using Test

using ArnoldiMethod
using LinearAlgebra

@testset "Zero eigenvalues & low-rank matrices" begin
    # Construct a rank-3 matrix, then with 100% change it'll find the first
    # three eigenvectors in 3 iterations. From then on, all random vectors
    # form a basis for the multiple eigenvalue -- for now we just test that
    # the stopping criterion works for these zero eigenvalues, we don't take
    # into account that it finds the complete eigenspace of the 0 eigenvalue.
    # (we have too few basis vectors for it anyways)

    A = rand(10, 3)

    # Rank 3 matrix.
    B = A * A' 

    schur, history = partial_schur(B, nev = 5, min = 5, max = 7, tol = eps())

    @test history.converged
    @test history.mvproducts == 7
    @test norm(schur.Q'schur.Q - I) < 100eps()
    @test norm(B * schur.Q - schur.Q * schur.R) < 100eps()
    @test norm(diag(schur.R)[4:7]) < 100eps()
end